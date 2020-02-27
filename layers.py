from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python.ops import array_ops, init_ops, math_ops, rnn_cell_impl, gen_array_ops
from tensorflow.python.platform import tf_logging as logging


class CoupledInputForgetGateLSTMCell(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, use_peepholes=False,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=1, num_proj_shards=1,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=math_ops.tanh, reuse=None):
        super(CoupledInputForgetGateLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be"
                " deprecated. use state_is_tuple=True.", self
            )
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._reuse = reuse

        if num_proj:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
                                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units)
                                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape("[-1])

        self.w_xi = tf.get_variable("_w_xi", [input_size.value, self._num_units])
        self.w_hi = tf.get_variable("_w_hi", [self._num_units, self._num_units])
        self.w_ci = tf.get_variable("_w_ci", [self._num_units, self._num_units])

        self.w_xo = tf.get_variable("_w_xo", [input_size.value, self._num_units])
        self.w_ho = tf.get_variable("_w_ho", [self._num_units, self._num_units])
        self.w_co = tf.get_variable("_w_co", [self._num_units, self._num_units])

        self.w_xc = tf.get_variable("_w_xc", [input_size.value, self._num_units])
        self.w_hc = tf.get_variable("_w_hc", [self._num_units, self._num_units])

        self.b_i = tf.get_variable("_b_i", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_c = tf.get_variable("_b_c", [self._num_units], initializer=init_ops.zeros_initializer())
        self.b_o = tf.get_variable("_b_o", [self._num_units], initializer=init_ops.zeros_initializer())

        i_t = sigmoid(math_ops.matmul(inputs, self.w_xi) +
                      math_ops.matmul(m_prev, self.w_hi) +
                      math_ops.matmul(c_prev, self.w_ci) +
                      self.b_i)

        c_t = ((1 - i_t) * c_prev + i_t * self._activation(math_ops.matmul(inputs, self.w_xc) +
                                                           math_ops.matmul(m_prev, self.w_hc) + self.b_c))

        o_t = sigmoid(math_ops.matmul(inputs, self.w_xo) +
                      math_ops.matmul(m_prev, self.w_ho) +
                      math_ops.matmul(c_t, self.w_co) +
                      self.b_o)

        h_t = o_t * self._activation(c_t)
        new_state = (rnn_cell_impl.LSTMStateTuple(c_t, h_t) if self._state_is_tuple else
                     array_ops.concat([c_t, h_t], 1))
        return h_t, new_state


def casual_dilated_conv(value, filters, rate, padding, name=None):
    with ops.name_scope(name, "atrous_conv2d", [value, filters]) as name:
        value = ops.convert_to_tensor(value, name="value")
        filters = ops.convert_n_to_tensor(filters, name="filters")
        if not value.get_shape()[3].is_compatible_with(filters.get_shape()[2]):
            raise ValueError("")
        if rate < 1:
            raise ValueError("")
        if rate == 1:
            value = tf.nn.conv2d(input=value,
                                 filter=filters,
                                 strides=[1, 1, 1, 1],
                                 padding=padding)
            return value
        if padding == "SAME":
            if filters.get_shape().is_fully_defined():
                filter_shape = filters.get_shape().as_list()
            else:
                filter_shape = array_ops.shape(filters)
            filter_width = filter_shape[1]

            filter_width_up = filter_width + (filter_width - 1) * (rate - 1)
            pad_width = filter_width_up - 1

            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        elif padding == "VALID":
            pad_left = 0
            pad_right = 0
        else:
            raise ValueError("")

        if value.get_shape().is_fully_defined():
            value_shape = value.get_shape().as_list()
        else:
            value_shape = array_ops.shape(value)

        in_width = value_shape[2] + pad_left + pad_right

        pad_right_extra = (rate - in_width % rate) % rate

        space_to_batch_pad = [[0, 0], [pad_left, pad_right + pad_right_extra]]

        value = gen_array_ops.space_to_batch_nd(input=value,
                                                paddings=space_to_batch_pad,
                                                block_shape=[1, rate])

        value = tf.nn.conv2d(input=value,
                             filters=filters,
                             strides=[1, 1, 1, 1],
                             padding="VALID",
                             name=name)

        batch_to_space_crop = [[0, 0], [0, pad_right_extra]]

        value = gen_array_ops.batch_to_space_nd(input=value,
                                                crops=batch_to_space_crop,
                                                block_shape=[1, rate])
        return value
