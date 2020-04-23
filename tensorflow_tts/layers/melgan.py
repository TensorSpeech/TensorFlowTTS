# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Layer modules for Melgan."""

import tensorflow as tf
from tensorflow_tts.utils import WeightNormalization


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], self.padding_type)


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 is_weight_norm,
                 initializer_range,
                 **kwargs):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_range)
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class TFResidualStack(tf.keras.layers.Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size,
                 filters,
                 dilation_rate,
                 use_bias,
                 nonlinear_activation,
                 nonlinear_activation_params,
                 is_weight_norm,
                 initializer_range,
                 **kwargs):
        """Initialize TFResidualStack module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (int): Dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__(**kwargs)
        self.blocks = [
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_range)
            ),
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params),
            tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=1,
                                   use_bias=use_bias,
                                   kernel_initializer=get_initializer(initializer_range))
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=filters,
                                               kernel_size=1,
                                               use_bias=use_bias,
                                               kernel_initializer=get_initializer(initializer_range),
                                               name='shortcut')

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks)
            self.shortcut = WeightNormalization(self.shortcut)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers"""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass
