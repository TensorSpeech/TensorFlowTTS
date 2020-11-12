# -*- coding: utf-8 -*-
# Copyright 2020 The TensorFlowTTS Team and Tomoki Hayashi (@kan-bayashi)
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

"""Parallel-wavegan Modules. Based on pytorch implementation (https://github.com/kan-bayashi/ParallelWaveGAN)"""

import tensorflow as tf


def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.he_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        HeNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.he_normal(seed=initializer_seed)


class TFConv1d1x1(tf.keras.layers.Conv1D):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, filters, use_bias, padding, initializer_seed, **kwargs):
        """Initialize 1x1 Conv1d module."""
        super().__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            dilation_rate=1,
            use_bias=use_bias,
            kernel_initializer=get_initializer(initializer_seed),
            **kwargs,
        )


class TFConv1d(tf.keras.layers.Conv1D):
    """Conv1d with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        initializer_seed = kwargs.pop("initializer_seed", 42)
        super().__init__(
            *args, **kwargs, kernel_initializer=get_initializer(initializer_seed)
        )


class TFResidualBlock(tf.keras.layers.Layer):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        dropout_rate=0.0,
        dilation_rate=1,
        use_bias=True,
        use_causal_conv=False,
        initializer_seed=42,
        **kwargs,
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout_rate (float): Dropout probability.
            dilation_rate (int): Dilation factor.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.
            initializer_seed (int32): initializer seed.
        """
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        # no future time stamps available
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = TFConv1d(
            filters=gate_channels,
            kernel_size=kernel_size,
            padding="same" if self.use_causal_conv is False else "causal",
            strides=1,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            initializer_seed=initializer_seed,
        )

        # local conditionong
        if aux_channels > 0:
            self.conv1x1_aux = TFConv1d1x1(
                gate_channels,
                use_bias=False,
                padding="same",
                initializer_seed=initializer_seed,
                name="conv1x1_aux",
            )
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = TFConv1d1x1(
            residual_channels,
            use_bias=use_bias,
            padding="same",
            initializer_seed=initializer_seed,
            name="conv1x1_out",
        )
        self.conv1x1_skip = TFConv1d1x1(
            skip_channels,
            use_bias=use_bias,
            padding="same",
            initializer_seed=initializer_seed,
            name="conv1x1_skip",
        )

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, x, c, training=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, T, residual_channels).
            Tensor: Output tensor for skip connection (B, T, skip_channels).
        """
        residual = x
        x = self.dropout(x, training=training)
        x = self.conv(x)

        # split into two part for gated activation
        xa, xb = tf.split(x, 2, axis=-1)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = tf.split(c, 2, axis=-1)
            xa, xb = xa + ca, xb + cb

        x = tf.nn.tanh(xa) * tf.nn.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = self.conv1x1_out(x)
        x = (x + residual) * tf.math.sqrt(0.5)

        return x, s


class TFStretch1d(tf.keras.layers.Layer):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, method="nearest", **kwargs):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            method (str): Interpolation method.

        """
        super().__init__(**kwargs)
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.method = method

    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, T, C, 1).
        Returns:
            Tensor: Interpolated tensor (B, T * x_scale, C * y_scale, 1)

        """
        x_shape = tf.shape(x)
        new_size = (x_shape[1] * self.x_scale, x_shape[2] * self.y_scale)
        x = tf.image.resize(x, method=self.method, size=new_size)
        return x


class TFUpsampleNetWork(tf.keras.layers.Layer):
    """Upsampling network module."""

    def __init__(
        self,
        output_channels,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        use_causal_conv=False,
        **kwargs,
    ):
        """Initialize upsampling network module.

        Args:
            output_channels (int): output feature channels.
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super().__init__(**kwargs)
        self.use_causal_conv = use_causal_conv
        self.up_layers = []

        for scale in upsample_scales:
            # interpolation layer
            stretch = TFStretch1d(
                scale, 1, interpolate_mode, name="stretch_._{}".format(scale)
            )  # ->> outputs: [B, T * scale, C * 1, 1]
            self.up_layers += [stretch]

            # conv layer
            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            kernel_size = scale * 2 + 1
            conv = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(kernel_size, freq_axis_kernel_size),
                padding="causal" if self.use_causal_conv is True else "same",
                use_bias=False,
            )  # ->> outputs: [B, T * scale, C * 1, 1]
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(tf.keras.layers, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    def call(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, T, C).
        Returns:
            Tensor: Upsampled tensor (B, T', C), where T' = T * prod(upsample_scales).
        """
        c = tf.expand_dims(c, -1)  # [B, T, C, 1]
        for f in self.up_layers:
            c = f(c)
        return tf.squeeze(c, -1)  # [B, T, C]


class TFConvInUpsampleNetWork(tf.keras.layers.Layer):
    """Convolution + upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode="nearest",
        freq_axis_kernel_size=1,
        aux_channels=80,
        aux_context_window=0,
        use_causal_conv=False,
        initializer_seed=42,
        **kwargs,
    ):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.

        """
        super().__init__(**kwargs)
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0

        # To capture wide-context information in conditional features
        kernel_size = (
            aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        )

        self.conv_in = TFConv1d(
            filters=aux_channels,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            initializer_seed=initializer_seed,
            name="conv_in",
        )
        self.upsample = TFUpsampleNetWork(
            output_channels=aux_channels,
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
            name="upsample_network",
        )

    def call(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, T', C).
    
        Returns:
            Tensor: Upsampled tensor (B, T, C),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).

        Note:
            The length of inputs considers the context window size.
        """
        c_ = self.conv_in(c)
        return self.upsample(c_)


class TFParallelWaveGANGenerator(tf.keras.Model):
    """Parallel WaveGAN Generator module."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = config.out_channels
        self.aux_channels = config.aux_channels
        self.n_layers = config.n_layers
        self.stacks = config.stacks
        self.kernel_size = config.kernel_size
        self.upsample_params = config.upsample_params

        # check the number of layers and stacks
        assert self.n_layers % self.stacks == 0
        n_layers_per_stack = self.n_layers // self.stacks

        # define first convolution
        self.first_conv = TFConv1d1x1(
            filters=config.residual_channels,
            use_bias=True,
            padding="same",
            initializer_seed=config.initializer_seed,
            name="first_convolution",
        )

        # define conv + upsampling network
        if config.upsample_conditional_features:
            self.upsample_params.update({"use_causal_conv": config.use_causal_conv})
            self.upsample_params.update(
                {
                    "aux_channels": config.aux_channels,
                    "aux_context_window": config.aux_context_window,
                }
            )
            self.upsample_net = TFConvInUpsampleNetWork(**self.upsample_params)
        else:
            self.upsample_net = None

        # define residual blocks
        self.conv_layers = []
        for layer in range(self.n_layers):
            dilation_rate = 2 ** (layer % n_layers_per_stack)
            conv = TFResidualBlock(
                kernel_size=config.kernel_size,
                residual_channels=config.residual_channels,
                gate_channels=config.gate_channels,
                skip_channels=config.skip_channels,
                aux_channels=config.aux_channels,
                dilation_rate=dilation_rate,
                dropout_rate=config.dropout_rate,
                use_bias=config.use_bias,
                use_causal_conv=config.use_causal_conv,
                initializer_seed=config.initializer_seed,
                name="residual_block_._{}".format(layer),
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = [
            tf.keras.layers.ReLU(),
            TFConv1d1x1(
                filters=config.skip_channels,
                use_bias=config.use_bias,
                padding="same",
                initializer_seed=config.initializer_seed,
            ),
            tf.keras.layers.ReLU(),
            TFConv1d1x1(
                filters=config.out_channels,
                use_bias=True,
                padding="same",
                initializer_seed=config.initializer_seed,
            ),
            tf.keras.layers.Activation("tanh"),
        ]

    def _build(self):
        mels = tf.random.uniform(shape=[2, 20, 80], dtype=tf.float32)
        self(mels, training=tf.cast(True, tf.bool))

    def call(self, mels, training=False, **kwargs):
        """Calculate forward propagation.

        Args:
            mels (Tensor): Local conditioning auxiliary features (B, T', C).
        Returns:

            Tensor: Output tensor (B, T, 1)
        """
        # perform upsampling
        if mels is not None and self.upsample_net is not None:
            c = self.upsample_net(mels)

        # random noise x
        # enccode to hidden representation
        x = tf.expand_dims(tf.random.normal(shape=tf.shape(c)[0:2]), axis=2)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, training=training)
            skips += h
        skips *= tf.math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels"),
        ],
    )
    def inference(self, mels):
        """Calculate forward propagation.

        Args:
            c (Tensor): Local conditioning auxiliary features (B, T', C).
        Returns:

            Tensor: Output tensor (B, T, 1)
        """
        # perform upsampling
        if mels is not None and self.upsample_net is not None:
            c = self.upsample_net(mels)

        # enccode to hidden representation
        x = tf.expand_dims(tf.random.normal(shape=tf.shape(c)[0:2]), axis=2)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, training=False)
            skips += h
        skips *= tf.math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x


class TFParallelWaveGANDiscriminator(tf.keras.Model):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        assert (config.kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert config.dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = []
        for i in range(config.n_layers - 1):
            if i == 0:
                dilation_rate = 1
            else:
                dilation_rate = (
                    i if config.dilation_factor == 1 else config.dilation_factor ** i
                )
            self.conv_layers += [
                TFConv1d(
                    filters=config.conv_channels,
                    kernel_size=config.kernel_size,
                    padding="same",
                    dilation_rate=dilation_rate,
                    use_bias=config.use_bias,
                    initializer_seed=config.initializer_seed,
                )
            ]
            self.conv_layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(
                    **config.nonlinear_activation_params
                )
            ]
        self.conv_layers += [
            TFConv1d(
                filters=config.out_channels,
                kernel_size=config.kernel_size,
                padding="same",
                use_bias=config.use_bias,
                initializer_seed=config.initializer_seed,
            )
        ]

        if config.apply_sigmoid_at_last:
            self.conv_layers += [
                tf.keras.layers.Activation("sigmoid"),
            ]

    def _build(self):
        x = tf.random.uniform(shape=[2, 16000, 1])
        self(x)

    def call(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, T, 1).

        Returns:
            Tensor: Output tensor (B, T, 1)
        """
        for f in self.conv_layers:
            x = f(x)
        return x
