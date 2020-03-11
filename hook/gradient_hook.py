import tensorflow as tf

from tensorflow.estimator import SessionRunHook


class AdversarialTrainingHelper:
    """ 用于存储梯度 """

    def __init__(self):
        self.is_adv_step = False
        self.grads_vars = None
        self.perturb = None

    def save_grads(self, loss):
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        # This is how the model was pre-trained.
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        grads_vars = {v: g for g, v in zip(grads, tvars)}
        # 如果是正常训练的话，则需要保存梯度信息
        if not self.is_adv_step:
            self.grads_vars = grads_vars
        else:
            self.grads_vars = None

        # 切换步伐
        self.is_adv_step = not self.is_adv_step
        return grads_vars

    # Adds gradient to embedding and recomputes classification loss.
    def _scale_l2(self, x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def compute_embedding_perturb(self, loss, word_embedding):
        if not self.is_adv_step:
            grad, = tf.gradients(
                loss,
                word_embedding)
            grad = tf.stop_gradient(grad)
            self.perturb = self._scale_l2(grad, 0.125)  # set low for tpu mode   [5, 384, 128]
        else:
            self.perturb = None
