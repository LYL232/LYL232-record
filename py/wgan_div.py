import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf

    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np


class DiscriminatorTrainLayer(Layer):
    def __init__(self, k, p, d_model: Model, g_model: Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.g_model = g_model
        self.k = k
        self.p = p

    def call(self, inputs, **kwargs):
        img_inputs, label_inputs, mask_inputs = inputs

        real_packed_imgs = tf.concat([img_inputs, label_inputs], axis=-1)

        x_fake = self.g_model([img_inputs, mask_inputs])

        fake_packed_imgs = tf.concat([img_inputs, x_fake], axis=-1)

        with tf.GradientTape() as tape:
            tape.watch(real_packed_imgs)
            tape.watch(fake_packed_imgs)
            x_real_score = self.d_model(real_packed_imgs)
            x_fake_score = self.d_model(fake_packed_imgs)
            real_grad, fake_grad = tape.gradient(
                [x_real_score, x_fake_score],
                [real_packed_imgs, fake_packed_imgs]
            )

        return x_real_score, x_fake_score, real_grad, fake_grad


class Generator(Model):
    """
    生成器模型，没有特殊限制
    """


class Critic(Model):
    """
    判别器模型，输入不限，输出一个标量，输出层不用激活函数，
    不适用BatchNormalization，激活可选LeakyRelu
    """


class WGANDiv:
    def __init__(
            self,
            img_rows, img_cols, channels,
            k=2, p=6,
            lr=2e-4
    ):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.lr = lr
        # Following parameter and optimizer set as recommended in paper
        self.k = k
        self.p = p

        # 生成器
        self.g_model = Generator()

        self.d_model = Critic()

        self.g_train_model = self.build_g_trainer()

        self.d_train_model = self.build_d_trainer()

        # 检查模型结构
        self.g_model.summary()
        self.d_model.summary()

    def build_d_trainer(self):
        # 整合模型（训练判别器）
        self.g_model.trainable = False
        self.d_model.trainable = True
        img_inputs = Input(shape=(self.img_rows, self.img_cols, self.channels))
        label_inputs = Input(shape=(self.img_rows, self.img_cols, 1))
        mask_inputs = Input(shape=(self.img_rows, self.img_cols, 1))

        x_real_score, x_fake_score, real_grad, fake_grad = \
            DiscriminatorTrainLayer(
                k=self.k, p=self.p,
                g_model=self.g_model, d_model=self.d_model
            )([img_inputs, label_inputs, mask_inputs])

        d_train_model = Model(
            inputs=[img_inputs, label_inputs, mask_inputs],
            outputs=(x_real_score, x_fake_score),
            name='discriminator-trainer'
        )

        d_loss = backend.mean(x_real_score - x_fake_score)
        real_grad_norm = backend.sum(real_grad ** 2, axis=[1, 2, 3]) ** (
                self.p / 2)
        fake_grad_norm = backend.sum(fake_grad ** 2, axis=[1, 2, 3]) ** (
                self.p / 2)
        grad_loss = backend.mean(real_grad_norm + fake_grad_norm) * self.k / 2

        d_train_model.add_loss(d_loss + grad_loss)
        d_train_model.add_metric(d_loss, aggregation='mean', name='d_loss')
        d_train_model.add_metric(
            grad_loss, aggregation='mean', name='grad_loss')
        d_train_model.compile(optimizer=Adam(self.lr, 0.5, clipvalue=100))

        return d_train_model

    def build_g_trainer(self):
        # 整合模型（训练生成器）
        self.g_model.trainable = True
        self.d_model.trainable = False
        img_inputs = Input(shape=(self.img_rows, self.img_cols, self.channels))
        mask_inputs = Input(shape=(self.img_rows, self.img_cols, 1))

        x_fake = self.g_model([img_inputs, mask_inputs])

        fake_packed_imgs = tf.concat([img_inputs, x_fake], axis=-1)
        fake_score = self.d_model(fake_packed_imgs)
        g_loss = backend.mean(fake_score)

        g_train_model = Model(
            inputs=(img_inputs, mask_inputs),
            outputs=(x_fake, fake_score),
            name='generator-trainer'
        )

        g_train_model.add_loss(g_loss)
        g_train_model.add_metric(
            g_loss, aggregation='mean', name='g_loss')
        g_train_model.compile(
            # 可以加一些loss函数作为辅助
            # loss=[tf.keras.losses.binary_crossentropy, None],
            optimizer=Adam(self.lr, 0.5, clipvalue=100)
        )

        return g_train_model

    def train(
            self, train_data_generator,
            validate_imgs: np.ndarray,
            plot_path: str, epochs,
            sample_interval=10,
            log_file='train-log.json'
    ):
        with open(log_file, 'w', encoding='utf8') as file:
            for epoch in range(epochs):
                log = {
                    'epoch': epoch,
                    'd_loss': 0, 'grad_loss': 0
                }

                self.g_model.trainable = False
                self.d_model.trainable = True

                imgs, labels, masks = next(train_data_generator)

                # Train the critic
                d_log = self.d_train_model.train_on_batch(
                    [imgs, labels, masks]
                )

                label_with_mask = np.concatenate([labels, masks], axis=3)

                self.g_model.trainable = True
                self.d_model.trainable = False
                g_log = self.g_train_model.train_on_batch(
                    [imgs, masks], [label_with_mask, None]
                )

                log['g_tot_loss'] = '%.4g' % g_log[0]
                log['bce'] = '%.4g' % g_log[2]
                log['acc'] = '%.4g' % g_log[3]
                log['g_loss'] = '%.4g' % g_log[4]
                log['d_loss'] = '%.4g' % d_log[1]
                log['grad_loss'] = '%.4g' % d_log[2]

                log_str = json.dumps(log)
                print(log_str)
                file.write(log_str + '\n')
                if epoch % sample_interval == 0:
                    self.validate(
                        epoch=epoch,
                        validate_imgs=validate_imgs,
                        plot_path=plot_path,
                    )
                    self.g_model.save_weights('./g-model-latest.hdf5')

    def validate(
            self, epoch,
            validate_imgs,
            plot_path
    ):
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # 画图：4行5列
        r, c = 4, 5

        gen_imgs = self.g_model.predict(validate_imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(32, 25.6))
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f'{plot_path}/epoch_{epoch}.png')
        plt.close()
