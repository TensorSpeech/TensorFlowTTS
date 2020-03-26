# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Layer modules for Melgan."""

import tensorflow as tf

from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import layers as tflayers


class TFReflectionPad1d(Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT"):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super(TFReflectionPad1d, self).__init__()
        self.padding_size = padding_size
        self.padding_type = padding_type

    @tf.function
    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], self.padding_type)


class TFConvTranspose1d(Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(self, filters, kernel_size, strides, padding):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super(TFConvTranspose1d, self).__init__()
        self.conv1d_transpose = tflayers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding
        )

    @tf.function
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


class TFResidualStack(Layer):
    """Tensorflow ResidualStack module."""

    def __init__(self,
                 kernel_size,
                 filters,
                 dilation_rate,
                 use_bias,
                 nonlinear_activation,
                 nonlinear_activation_params):
        """Initialize TFResidualStack module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (int): Dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super(TFResidualStack, self).__init__()
        self.blocks = [
            getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params),
            TFReflectionPad1d(dilation_rate),
            tflayers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                padding='valid'
            ),
            getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params),
            tflayers.Conv1D(filters=filters, kernel_size=1, use_bias=use_bias)
        ]
        self.shortcut = tflayers.Conv1D(filters=filters, kernel_size=1, use_bias=use_bias)

    @tf.function
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
