# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import logging

import numpy as np

import tensorflow as tf

from tensorflow_tts.layers import TFReflectionPad1d
from tensorflow_tts.layers import TFConvTranspose1d
from tensorflow_tts.layers import TFResidualStack

from tensorflow_tts.utils import WeightNormalization


class TFMelGANGenerator(tf.keras.Model):
    """Tensorflow MelGAN generator module."""

    def __init__(self, config, **kwargs):
        """Initialize TFMelGANGenerator module.

        Args:
            config: config object of Melgan generator.
        """
        super().__init__(**kwargs)

        # check hyper parameter is valid or not
        assert config.filters >= np.prod(config.upsample_scales)
        assert config.filters % (2 ** len(config.upsample_scales)) == 0

        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='first_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.filters,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias)
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding='same',
                    is_weight_norm=config.is_weight_norm,
                    name='conv_transpose_._{}'.format(i)
                )
            ]

            # ad residual stack layer
            for j in range(config.stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=config.stack_kernel_size,
                        filters=config.filters // (2 ** (i + 1)),
                        dilation_rate=config.stack_kernel_size ** j,
                        use_bias=config.use_bias,
                        nonlinear_activation=config.nonlinear_activation,
                        nonlinear_activation_params=config.nonlinear_activation_params,
                        is_weight_norm=config.is_weight_norm,
                        name='residual_stack_._{}._._{}'.format(i, j)
                    )
                ]
        # add final layer
        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(**config.nonlinear_activation_params),
            TFReflectionPad1d((config.kernel_size - 1) // 2,
                              padding_type=config.padding_type,
                              name='last_reflect_padding'),
            tf.keras.layers.Conv1D(filters=config.out_channels,
                                   kernel_size=config.kernel_size,
                                   use_bias=config.use_bias)
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh")]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.melgan = tf.keras.models.Sequential(layers)

    def call(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, T, channels)

        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)

        """
        return self.melgan(c)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers"""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].__name__.lower()
                if "conv" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFMelGANDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(self,
                 out_channels=1,
                 kernel_sizes=[5, 3],
                 filters=16,
                 max_downsample_filters=1024,
                 use_bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 padding_type="REFLECT",
                 is_weight_norm=True, **kwargs):
        """Initilize MelGAN discriminator module.
        Args:
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            filters (int): Initial number of filters for conv layer.
            max_downsample_filters (int): Maximum number of filters for downsampling layers.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
        """
        super().__init__(**kwargs)
        discriminator = []

        # check kernel_size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        in_chs = filters
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_filters)
            discriminator += [
                tf.keras.layers.Conv1D(
                    filters=out_chs,
                    kernel_size=downsample_scale * 10 + 1,
                    strides=downsample_scale,
                    padding='same',
                    use_bias=use_bias,
                )
            ]
            discriminator += [
                getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
            ]
            in_chs = out_channels

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_chs,
                kernel_size=kernel_sizes[0],
                padding='same',
                use_bias=use_bias
            )
        ]
        discriminator += [
            getattr(tf.keras.layers, nonlinear_activation)(**nonlinear_activation_params)
        ]
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_channels,
                kernel_size=kernel_sizes[1],
                padding='same',
                use_bias=use_bias
            )
        ]

        if is_weight_norm is True:
            self._apply_weightnorm(discriminator)

        self.disciminator = discriminator

    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of output tensors of each layer.

        """
        outs = []
        for f in self.disciminator:
            x = f(x)
            outs += [x]
        return outs

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers"""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].__name__.lower()
                if "conv" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFMelGANMultiScaleDiscriminator(tf.keras.Model):
    """MelGAN multi-scale discriminator module."""

    def __init__(self, config, **kwargs):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            config: config object for melgan discriminator
        """
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(config.scales):
            self.discriminator += [
                TFMelGANDiscriminator(
                    out_channels=config.out_channels,
                    kernel_sizes=config.kernel_sizes,
                    filters=config.filters,
                    max_downsample_filters=config.max_downsample_filters,
                    use_bias=config.use_bias,
                    downsample_scales=config.downsample_scales,
                    nonlinear_activation=config.nonlinear_activation,
                    nonlinear_activation_params=config.nonlinear_activation_params,
                    padding_type=config.padding_type,
                    is_weight_norm=config.is_weight_norm,
                    name='melgan_discriminator_scale_._{}'.format(i)
                )
            ]
            self.pooling = getattr(tf.keras.layers, config.downsample_pooling)(**config.downsample_pooling_params)

    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
            x = self.pooling(x)
        return outs
