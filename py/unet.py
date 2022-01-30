if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf

    for _ in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(_, True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, \
    UpSampling2D, concatenate, Activation, BatchNormalization
import tensorflow as tf


class UnetDownSamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            dropout: float = 0.0,
            activation='relu',
            pooling=True,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation

        self.conv1 = None
        self.bn1 = None
        self.act1 = None

        self.conv2 = None
        self.bn2 = None
        self.act2 = None

        self.dropout_layer = None
        self.pooling_layer = None

    def build(self, input_shape):
        self.conv1 = Conv2D(
            self.filters, 3, padding='same',
            kernel_initializer='he_normal',
            input_shape=input_shape, use_bias=False
        )
        self.bn1 = BatchNormalization(scale=False, axis=3)
        if isinstance(self.activation, str):
            self.act1 = Activation(self.activation)
        elif callable(self.activation):
            self.act1 = self.activation
        self.conv2 = Conv2D(
            self.filters, 3, padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.bn2 = BatchNormalization(scale=False, axis=3)
        if isinstance(self.activation, str):
            self.act2 = Activation(self.activation)
        elif callable(self.activation):
            self.act2 = self.activation
        if self.dropout > 1e-6:
            self.dropout_layer = Dropout(self.dropout)
        if self.pooling:
            self.pooling_layer = MaxPooling2D(pool_size=(2, 2))
        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        res = self.conv1(inputs)
        res = self.bn1(res)
        if self.act1 is not None:
            res = self.act1(res)

        res = self.conv2(res)
        res = self.bn2(res)
        if self.act2 is not None:
            res = self.act2(res)

        if self.dropout_layer is not None:
            res = self.dropout_layer(res)

        if self.pooling_layer is not None:
            return res, self.pooling_layer(res)
        return res


class UnetUpSamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            activation='relu',
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.activation = activation

        self.up = None
        self.conv1 = None
        self.bn1 = None
        self.act1 = None

        self.conv2 = None
        self.bn2 = None
        self.act2 = None

        self.conv3 = None
        self.bn3 = None
        self.act3 = None

    def build(self, input_shape):
        self.up = UpSampling2D(size=(2, 2))

        self.conv1 = Conv2D(
            self.filters, 2,
            padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.bn1 = BatchNormalization(scale=False, axis=3)
        if isinstance(self.activation, str):
            self.act1 = Activation(self.activation)
        elif callable(self.activation):
            self.act1 = self.activation

        self.conv2 = Conv2D(
            self.filters, 3,
            padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.bn2 = BatchNormalization(scale=False, axis=3)
        if isinstance(self.activation, str):
            self.act2 = Activation(self.activation)
        elif callable(self.activation):
            self.act2 = self.activation

        self.conv3 = Conv2D(
            self.filters, 3,
            padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.bn3 = BatchNormalization(scale=False, axis=3)
        if isinstance(self.activation, str):
            self.act3 = Activation(self.activation)
        elif callable(self.activation):
            self.act3 = self.activation

        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        low, crop = inputs

        res = self.up(low)

        res = self.conv1(res)
        res = self.bn1(res)
        if self.act1 is not None:
            res = self.act1(res)

        res = concatenate([crop, res], axis=3)

        res = self.conv2(res)
        res = self.bn2(res)
        if self.act2 is not None:
            res = self.act2(res)

        res = self.conv3(res)
        res = self.bn3(res)
        if self.act3 is not None:
            res = self.act3(res)

        return res


class Unet(Model):
    def __init__(
            self,
            base_filters: int = 64,
            activation='relu',
            output_channels: int = 1,
            output_activation='tanh',
            layers: int = 4,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.left_layers = [
            UnetDownSamplingLayer(
                filters=base_filters * (2 ** i),
                dropout=0.5 if i >= 3 else 0.0,
                pooling=True,
                activation=activation
            ) for i in range(layers)
        ]
        self.mid_layer = UnetDownSamplingLayer(
            filters=base_filters * (2 ** layers),
            dropout=0.5, pooling=False,
            activation=activation
        )
        self.right_layers = [
            UnetUpSamplingLayer(
                filters=base_filters * (2 ** (layers - 1 - i)),
                activation=activation,
            ) for i in range(layers)
        ]
        self.output_layer = Conv2D(
            output_channels, 3,
            padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.output_bn = BatchNormalization(scale=False, axis=3)
        if isinstance(output_activation, str):
            self.output_activation = Activation(output_activation)
        elif callable(output_activation):
            self.output_activation = output_activation
        else:
            self.output_activation = None

    def call(self, inputs, training=None, mask=None):
        crops = []
        res = inputs
        for layer in self.left_layers:
            crop, res = layer(res)
            crops.append(crop)
        res = self.mid_layer(res)
        for i, layer in enumerate(self.right_layers):
            res = layer([res, crops[-1 * (i + 1)]])
        res = self.output_layer(res)
        res = self.output_bn(res)
        if self.output_activation is not None:
            res = self.output_activation(res)
        return res


def main():
    import numpy as np
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = Unet(
        base_filters=64,
        activation='relu',
        output_channels=3,
        output_activation='sigmoid',
        layers=4
    )
    model.build(input_shape=x_train.shape)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.binary_crossentropy
    )
    model.fit(
        x_train,
        x_train,
        verbose=1,
        epochs=5
    )
    res = model.predict(x_test)
    np.save('predict.npy', res)


if __name__ == '__main__':
    main()
