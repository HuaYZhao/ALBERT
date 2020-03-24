# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

from contrib import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from contrib import batching


def variable(name,
             shape=None,
             dtype=None,
             initializer=None,
             regularizer=None,
             trainable=True,
             collections=None,
             caching_device=None,
             device=None,
             partitioner=None,
             custom_getter=None,
             use_resource=None,
             synchronization=variables.VariableSynchronization.AUTO,
             aggregation=variables.VariableAggregation.NONE):
    """Gets an existing variable with these parameters or creates a new one.

    Args:
      name: the name of the new or existing variable.
      shape: shape of the new or existing variable.
      dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: initializer for the variable if one is created.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: A list of collection names to which the Variable will be added.
        If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's device.
      device: Optional device to place the variable. It can be an string or a
        function that is called to get the device for the variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      custom_getter: Callable that allows overwriting the internal get_variable
        method and has to have the same signature.
      use_resource: If `True` use a ResourceVariable instead of a Variable.
      synchronization: Indicates when a distributed a variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      The created or existing variable.
    """
    collections = list(collections if collections is not None else
                       [ops.GraphKeys.GLOBAL_VARIABLES])

    # Remove duplicates
    collections = list(set(collections))
    getter = variable_scope.get_variable
    if custom_getter is not None:
        getter = functools.partial(
            custom_getter, reuse=variable_scope.get_variable_scope().reuse)
    with ops.device(device or ''):
        return getter(
            name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=partitioner,
            use_resource=use_resource,
            synchronization=synchronization,
            aggregation=aggregation)


def model_variable(name,
                   shape=None,
                   dtype=dtypes.float32,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   collections=None,
                   caching_device=None,
                   device=None,
                   partitioner=None,
                   custom_getter=None,
                   use_resource=None,
                   synchronization=variables.VariableSynchronization.AUTO,
                   aggregation=variables.VariableAggregation.NONE):
    """Gets an existing model variable with these parameters or creates a new one.

    Args:
      name: the name of the new or existing variable.
      shape: shape of the new or existing variable.
      dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: initializer for the variable if one is created.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: A list of collection names to which the Variable will be added.
        Note that the variable is always also added to the
        `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's device.
      device: Optional device to place the variable. It can be an string or a
        function that is called to get the device for the variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      custom_getter: Callable that allows overwriting the internal get_variable
        method and has to have the same signature.
      use_resource: If `True` use a ResourceVariable instead of a Variable.
      synchronization: Indicates when a distributed a variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

    Returns:
      The created or existing variable.
    """
    collections = list(collections or [])
    collections += [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
    var = variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        device=device,
        partitioner=partitioner,
        custom_getter=custom_getter,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)
    return var


def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
    """Adds a Layer Normalization layer.
    Based on the paper:
      "Layer Normalization"
      Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
      https://arxiv.org/abs/1607.06450.
    Can be used as a normalizer function for conv2d and fully_connected.
    Given a tensor `inputs` of rank `R`, moments are calculated and normalization
    is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
    if requested, is performed over axes `begin_params_axis .. R - 1`.
    By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
    meaning that normalization is performed over all but the first axis
    (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
    parameters are calculated for the rightmost axis (the `C` if `inputs` is
    `NHWC`).  Scaling and recentering is performed via broadcast of the
    `beta` and `gamma` parameters with the normalized tensor.
    The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
    and this part of the inputs' shape must be fully defined.
    Args:
      inputs: A tensor having rank `R`. The normalization is performed over axes
        `begin_norm_axis ... R - 1` and centering and scaling parameters are
        calculated over `begin_params_axis ... R - 1`.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used. When the
        next layer is linear (also e.g. `nn.relu`), this can be disabled since the
        scaling can be done by the next layer.
      activation_fn: Activation function, default set to None to skip it and
        maintain a linear activation.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional collections for the variables.
      outputs_collections: Collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      begin_norm_axis: The first normalization dimension: normalization will be
        performed along dimensions `begin_norm_axis : rank(inputs)`
      begin_params_axis: The first parameter (beta, gamma) dimension: scale and
        centering parameters will have dimensions
        `begin_params_axis : rank(inputs)` and will be broadcast with the
          normalized inputs accordingly.
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation, having the same
      shape and dtype as `inputs`.
    Raises:
      ValueError: If the rank of `inputs` is not known at graph build time,
        or if `inputs.shape[begin_params_axis:]` is not fully defined at
        graph build time.
    """
    with variable_scope.variable_scope(
            scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        inputs_shape = inputs.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                             'must be < rank(inputs) (%d)' %
                             (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                (inputs.name, begin_params_axis, inputs_shape))
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,
                                                              'beta')
            beta = model_variable(
                'beta',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.zeros_initializer(),
                collections=beta_collections,
                trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(
                variables_collections, 'gamma')
            gamma = model_variable(
                'gamma',
                shape=params_shape,
                dtype=dtype,
                initializer=init_ops.ones_initializer(),
                collections=gamma_collections,
                trainable=trainable)
        # By default, compute the moments across all the dimensions except the one with index 0.
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        # Note that epsilon must be increased for float16 due to the limited
        # representable range.
        variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
        outputs = nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=variance_epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None):
    """Fused implementation of `map` and `batch`.

    Maps `map_func` across `batch_size` consecutive elements of this dataset
    and then combines them into a batch. Functionally, it is equivalent to `map`
    followed by `batch`. However, by fusing the two transformations together, the
    implementation can be more efficient. Surfacing this transformation in the API
    is temporary. Once automatic input pipeline optimization is implemented,
    the fusing of `map` and `batch` will happen automatically and this API will be
    deprecated.

    Args:
      map_func: A function mapping a nested structure of tensors to another nested
        structure of tensors.
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number of batches to create in parallel. On one hand,
        higher values can help mitigate the effect of stragglers. On the other
        hand, higher values can increase contention if CPU is scarce.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in case its size is smaller than
        desired; the default behavior is not to drop the smaller batch.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number of elements to process in parallel. If not
        specified, `batch_size * num_parallel_batches` elements will be processed
        in parallel.

    Returns:
      A `Dataset` transformation function, which can be passed to
      `tf.data.Dataset.apply`.

    Raises:
      ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
        specified.
    """
    return batching.map_and_batch(map_func, batch_size, num_parallel_batches,
                                  drop_remainder, num_parallel_calls)
