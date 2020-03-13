import tensorflow as tf

from tensorflow.estimator import SessionRunHook
import numpy as np


class GlaceHook(SessionRunHook):

    def __init__(self, ops):
        self._ops = ops

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(self._ops)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        step = run_values.results[0]
        np.savez(f"save_{step}.npz", **run_values.results[1])
