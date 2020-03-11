import tensorflow as tf

from tensorflow.estimator import SessionRunHook


class GradientsHook(SessionRunHook):
    """ 用于存储梯度 """

    def __init__(self):
        self.last_norm_grads = None

    def begin(self):
        pass
