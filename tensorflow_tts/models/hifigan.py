# -*- coding: utf-8 -*-
# Copyright 2020 The Hifigan Authors and TensorflowTTS Team.
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
"""Hifi Modules."""

import numpy as np
import tensorflow as tf

from tensorflow_tts.models.melgan import TFReflectionPad1d
from tensorflow_tts.models.melgan import TFConvTranspose1d

from tensorflow_tts.utils import GroupConv1D
from tensorflow_tts.utils import WeightNormalization

from tensorflow_tts.models import TFMelGANGenerator


class TFHifiResBlock(tf.keras.layers.Layer):
    """Tensorflow Hifigan resblock 1 module."""

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
        """Initialize TFHifiResBlock module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (list): List dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            is_weight_norm (bool): Whether to use weight norm or not.
        """
        super().__init__(**kwargs)
        self.blocks_1 = []
        self.blocks_2 = []

        for i in range(len(dilation_rate)):
            self.blocks_1.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate[i]),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=dilation_rate[i],
                        use_bias=use_bias,
                    ),
                ]
            )
            self.blocks_2.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * 1),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=1,
                        use_bias=use_bias,
                    ),
                ]
            )

        self.activation = getattr(tf.keras.layers, nonlinear_activation)(
            **nonlinear_activation_params
        )

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks_1)
            self._apply_weightnorm(self.blocks_2)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        for c1, c2 in zip(self.blocks_1, self.blocks_2):
            xt = self.activation(x)
            for c in c1:
                xt = c(xt)
            xt = self.activation(xt)
            for c in c2:
                xt = c(xt)
            x = xt + x
        return x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFHifiGANGenerator(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # check hyper parameter is valid or not
        assert (
            config.stacks
            == len(config.stack_kernel_size)
            == len(config.stack_dilation_rate)
        )

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
                    TFHifiResBlock(
                        kernel_size=config.stack_kernel_size[j],
                        filters=config.filters // (2 ** (i + 1)),
                        dilation_rate=config.stack_dilation_rate[j],
                        use_bias=config.use_bias,
                        nonlinear_activation=config.nonlinear_activation,
                        nonlinear_activation_params=config.nonlinear_activation_params,
                        is_weight_norm=config.is_weight_norm,
                        initializer_seed=config.initializer_seed,
                        name="hifigan_resblock_._{}._._{}".format(i, j),
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
                dtype=tf.float32,
            ),
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.hifigan = tf.keras.models.Sequential(layers)

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
        return self.hifigan(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference_tflite(self, mels):
        return self.hifigan(mels)

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


class TFHifiGANPeriodDiscriminator(tf.keras.layers.Layer):
    """Tensorflow Hifigan period discriminator module."""

    def __init__(
        self,
        period,
        out_channels=1,
        n_layers=5,
        kernel_size=5,
        strides=3,
        filters=8,
        filter_scales=4,
        max_filters=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        initializer_seed=42,
        is_weight_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.period = period
        self.out_filters = out_channels
        self.convs = []

        for i in range(n_layers):
            self.convs.append(
                tf.keras.layers.Conv2D(
                    filters=min(filters * (filter_scales ** (i + 1)), max_filters),
                    kernel_size=(kernel_size, 1),
                    strides=(strides, 1),
                    padding="same",
                )
            )
        self.conv_post = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(3, 1), padding="same",
        )
        self.activation = getattr(tf.keras.layers, nonlinear_activation)(
            **nonlinear_activation_params
        )

        if is_weight_norm:
            self._apply_weightnorm(self.convs)
            self.conv_post = WeightNormalization(self.conv_post)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of output tensors.
        """
        shape = tf.shape(x)
        n_pad = tf.convert_to_tensor(0, dtype=tf.int32)
        if shape[1] % self.period != 0:
            n_pad = self.period - (shape[1] % self.period)
            x = tf.pad(x, [[0, 0], [0, n_pad], [0, 0]], "REFLECT")
        x = tf.reshape(
            x, [shape[0], (shape[1] + n_pad) // self.period, self.period, x.shape[2]]
        )
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
        x = self.conv_post(x)
        x = tf.reshape(x, [shape[0], -1, self.out_filters])
        return [x]

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFHifiGANMultiPeriodDiscriminator(tf.keras.Model):
    """Tensorflow Hifigan Multi Period discriminator module."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(len(config.period_scales)):
            self.discriminator += [
                TFHifiGANPeriodDiscriminator(
                    config.period_scales[i],
                    out_channels=config.out_channels,
                    n_layers=config.n_layers,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    filters=config.filters,
                    filter_scales=config.filter_scales,
                    max_filters=config.max_filters,
                    nonlinear_activation=config.nonlinear_activation,
                    nonlinear_activation_params=config.nonlinear_activation_params,
                    initializer_seed=config.initializer_seed,
                    is_weight_norm=config.is_weight_norm,
                    name="hifigan_period_discriminator_._{}".format(i),
                )
            ]

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: list of each discriminator outputs
        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
        return outs
