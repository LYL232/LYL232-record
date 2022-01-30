"""
在主脚本代码开头加入如下语句以限制tensorflow使用的GPU并设置
不一次性占满显存，在import tensorflow前执行才有效
"""
if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf

    for _ in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(_, True)
