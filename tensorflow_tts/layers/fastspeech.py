# -*- coding: utf-8 -*-

# Copyright 2020 TRINH LE (@l4zy9x)
#  MIT License (https://opensource.org/licenses/MIT)
"""Tensorflow Layer modules for FastSpeech."""
import tensorflow as tf
from tensorflow.python.keras import layers as tflayers


class TacotronEncoder(tflayers.Layer):
    "Tacotron encoder module"

    def __init__(self,
                 hidden_dim,
                 num_conv_layers,
                 conv_kernel_size,
                 conv_dilation_rate,
                 out_dim,
                 use_bias,
                 dropout):
        """Initialize TacotronEncoder module.
        Args:
            hidden_dim (int): Hidden dimension.
            num_conv_layers (int): Number of convolution layers.
            conv_kernel_size (int): Convolution kernel size
            conv_dilation_rate (int): Convolution dilation rate.
            out_dim: Output dimension
            use_bias (bool): Whether to add bias parameter.
            dropout (float): Dropout of all component, value in (0, 1)
        """
        super().__init__()
        assert hidden_dim % 2 == 0
        self.conv_blocks = [
            tflayers.Conv1D(filters=hidden_dim,
                            kernel_size=conv_kernel_size,
                            dilation_rate=conv_dilation_rate,
                            use_bias=use_bias,
                            padding='same'),
        ]

        self.bilstm = tflayers.Bidirectional(
            layer=tflayers.LSTM(hidden_dim // 2,
                                use_bias=use_bias,
                                return_sequences=True),
            backward_layer=tflayers.LSTM(hidden_dim // 2,
                                         use_bias=use_bias,
                                         go_backwards=True,
                                         return_sequences=True))

        self.linear = tflayers.Conv1D(filters=out_dim,
                                      kernel_size=1,
                                      use_bias=use_bias)

    @tf.function
    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.conv_blocks:
            _x = layer(_x)
        _x = self.bilstm(_x)
        y = self.linear(_x)

        return y


class Foo():
    def __init__(self):
        pass


if __name__ == "__main__":
    encoder = TacotronEncoder(128, 3, 3, None, 1, 64, True, 0.1)
    x = tf.random.uniform([4, 50, 128], dtype=tf.float32)
    encoder(x)
