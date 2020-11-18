# -*- coding: utf-8 -*-
# Copyright 2020 The MelGAN Authors and Minh Nguyen (@dathudeptrai)
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
"""MelGAN Modules."""

import numpy as np
import tensorflow as tf

from tensorflow_tts.utils import GroupConv1D, WeightNormalization


def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)


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
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0]],
            self.padding_type,
        )


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
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
            kernel_initializer=get_initializer(initializer_seed),
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

    def __init__(
        self,
        kernel_size,
        filters,
        dilation_rate,
        use_bias,
        nonlinear_activation,
        nonlinear_activation_params,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
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
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            ),
        ]
        self.shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            use_bias=use_bias,
            kernel_initializer=get_initializer(initializer_seed),
            name="shortcut",
        )

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
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


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
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="first_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.filters,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
                kernel_initializer=get_initializer(config.initializer_seed),
            ),
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(
                    **config.nonlinear_activation_params
                ),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding="same",
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name="conv_transpose_._{}".format(i),
                ),
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
                        initializer_seed=config.initializer_seed,
                        name="residual_stack_._{}._._{}".format(i, j),
                    )
                ]
        # add final layer
        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(
                **config.nonlinear_activation_params
            ),
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="last_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.out_channels,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
                kernel_initializer=get_initializer(config.initializer_seed),
                dtype=tf.float32,
            ),
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.melgan = tf.keras.models.Sequential(layers)

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference(self, mels):
        return self.melgan(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference_tflite(self, mels):
        return self.melgan(mels)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
        self(fake_mels)


class TFMelGANDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(
        self,
        out_channels=1,
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
        is_weight_norm=True,
        initializer_seed=0.02,
        **kwargs
    ):
        """Initilize MelGAN discriminator module.
        Args:
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15.
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
        discriminator = [
            TFReflectionPad1d(
                (np.prod(kernel_sizes) - 1) // 2, padding_type=padding_type
            ),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=int(np.prod(kernel_sizes)),
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
        ]

        # add downsample layers
        in_chs = filters
        with tf.keras.utils.CustomObjectScope({"GroupConv1D": GroupConv1D}):
            for downsample_scale in downsample_scales:
                out_chs = min(in_chs * downsample_scale, max_downsample_filters)
                discriminator += [
                    GroupConv1D(
                        filters=out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        strides=downsample_scale,
                        padding="same",
                        use_bias=use_bias,
                        groups=in_chs // 4,
                        kernel_initializer=get_initializer(initializer_seed),
                    )
                ]
                discriminator += [
                    getattr(tf.keras.layers, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                ]
                in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_chs,
                kernel_size=kernel_sizes[0],
                padding="same",
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            )
        ]
        discriminator += [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            )
        ]
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_channels,
                kernel_size=kernel_sizes[1],
                padding="same",
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            )
        ]

        if is_weight_norm is True:
            self._apply_weightnorm(discriminator)

        self.disciminator = discriminator

    def call(self, x, **kwargs):
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
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
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
                    initializer_seed=config.initializer_seed,
                    name="melgan_discriminator_scale_._{}".format(i),
                )
            ]
            self.pooling = getattr(tf.keras.layers, config.downsample_pooling)(
                **config.downsample_pooling_params
            )

    def call(self, x, **kwargs):
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
