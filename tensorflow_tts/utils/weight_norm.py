# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization as WeightNormalizationOriginal


class WeightNormalization(WeightNormalizationOriginal):
    """This class is modified from tensorlow_addons.layers.WeightNormalization
    But also support for convolution transpose.
    """

    def build(self, input_shape):
        """Build `Layer`"""
        # input_shape = tf.TensorShape(input_shape)
        # self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        # remove 2 lines above to run weight-norm on tf.function with dynamic shape

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1, if conv_traspose it's -2
        if "_transpose" in self.layer.name:
            self.layer_depth = int(kernel.shape[-2])
            self.kernel_norm_axes = list(range(kernel.shape.rank - 1))
        else:
            self.layer_depth = int(kernel.shape[-1])
            self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            dtype=kernel.dtype,
            trainable=True,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                self._naked_clone_layer = self.layer
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            if "_transpose" in self.layer.name:
                kernel = tf.nn.l2_normalize(tf.transpose(self.v, [0, 1, 3, 2]), axis=self.kernel_norm_axes) * g
                kernel = tf.transpose(kernel, perm=[0, 1, 3, 2])
            else:
                kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs
