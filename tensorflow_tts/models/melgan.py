# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import logging

import numpy as np

import tensorflow as tf
import tensorflow.python.keras.layers as tflayers

from tensorflow_tts.layers import TFReflectionPad1d
from tensorflow_tts.layers import TFConvTranspose1d
from tensorflow_tts.layers import TFResidualStack


class TFMelGANGenerator(tf.keras.Model):
    """Tensorflow MelGAN generator module."""

    def __init__(self,
                 out_channels=1,
                 kernel_size=7,
                 filters=512,
                 use_bias=True,
                 upsample_scales=[8, 8, 2, 2],
                 stack_kernel_size=3,
                 stacks=3,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 padding_type="REFLECT",
                 use_final_nolinear_activation=True
                 ):
        """Initialize TFMelGANGenerator module.

        Args:
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            filters (int): Initial number of channels for conv layer.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
            use_final_nolinear_activation (torch.nn.Module): Activation function for the final layer.

        """
        super(TFMelGANGenerator, self).__init__()

        # check hyper parameter is valid or not
        assert filters >= np.prod(upsample_scales)
        assert filters % (2 ** len(upsample_scales)) == 0

        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d((kernel_size - 1) // 2, padding_type=padding_type),
            tflayers.Conv1D(filters=filters, kernel_size=kernel_size, use_bias=use_bias)
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params),
                TFConvTranspose1d(
                    filters=filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding='same'
                )
            ]

            # ad residual stack layer
            for j in range(stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=stack_kernel_size,
                        filters=filters // (2 ** (i + 1)),
                        dilation_rate=stack_kernel_size ** j,
                        use_bias=use_bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        # add final layer
        layers += [
            getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params),
            TFReflectionPad1d((kernel_size - 1) // 2, padding_type=padding_type),
            tflayers.Conv1D(filters=out_channels, kernel_size=kernel_size, use_bias=use_bias)
        ]
        if use_final_nolinear_activation:
            layers += [tflayers.Activation("tanh")]

        self.melgan = tf.keras.models.Sequential(layers)

    @tf.function
    def call(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, T, channels)

        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)

        """
        return self.melgan(c)


class TFMelGANDiscriminator(tf.keras.Model):
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
                 padding_type="REFLECT"
                 ):
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
        super(TFMelGANDiscriminator, self).__init__()
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
                tflayers.Conv1D(
                    filters=out_chs,
                    kernel_size=downsample_scale * 10 + 1,
                    strides=downsample_scale,
                    padding='same',
                    use_bias=use_bias
                )
            ]
            discriminator += [
                getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params)
            ]
            in_chs = out_channels

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tflayers.Conv1D(
                filters=out_chs,
                kernel_size=kernel_sizes[0],
                padding='same',
                use_bias=use_bias
            )
        ]
        discriminator += [
            getattr(tflayers, nonlinear_activation)(**nonlinear_activation_params)
        ]
        discriminator += [
            tflayers.Conv1D(
                filters=out_channels,
                kernel_size=kernel_sizes[1],
                padding='same',
                use_bias=use_bias
            )
        ]
        self.disciminator = discriminator

    @tf.function
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


class TFMelGANMultiScaleDiscriminator(tf.keras.Model):
    """MelGAN multi-scale discriminator module."""

    def __init__(self,
                 out_channels=1,
                 scales=3,
                 downsample_pooling='AveragePooling1D',
                 downsample_pooling_params={
                     "pool_size": 4,
                     "strides": 2,
                 },
                 kernel_sizes=[5, 3],
                 filters=16,
                 max_downsample_filters=1024,
                 use_bias=True,
                 downsample_scales=[4, 4, 4, 4],
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"alpha": 0.2},
                 padding_type="REFLECT"
                 ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            filters (int): Initial number of channels for conv layer.
            max_downsample_filters (int): Maximum number of channels for downsampling layers.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")

        """
        super(TFMelGANMultiScaleDiscriminator, self).__init__()
        self.discriminator = []

        # add discriminator
        for _ in range(scales):
            self.discriminator += [
                TFMelGANDiscriminator(
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    filters=filters,
                    max_downsample_filters=max_downsample_filters,
                    use_bias=use_bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    padding_type=padding_type
                )
            ]
            self.pooling = getattr(tflayers, downsample_pooling)(**downsample_pooling_params)

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
