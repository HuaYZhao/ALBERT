import tensorflow as tf

from tensorflow.estimator import SessionRunHook
import numpy as np
import dill

i = 0


class GlaceHook(SessionRunHook):

    def __init__(self, ops):
        self._ops = ops

    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(self._ops)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        # np.savez(f"save_{run_values.results['global_step']}.npz", **run_values.results)
        global i
        dill.dump(run_values.results, open(f"save_{i}.pkl", 'wb'))
        i += 1

