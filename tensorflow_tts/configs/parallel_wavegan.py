# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
"""ParallelWaveGAN Config object."""


class ParallelWaveGANGeneratorConfig(object):
    """Initialize ParallelWaveGAN Generator Config."""

    def __init__(
        self,
        out_channels=1,
        kernel_size=3,
        n_layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        aux_context_window=2,
        dropout_rate=0.0,
        use_bias=True,
        use_causal_conv=False,
        upsample_conditional_features=True,
        upsample_params={"upsample_scales": [4, 4, 4, 4]},
        initializer_seed=42,
        **kwargs,
    ):
        """Init parameters for ParallelWaveGAN Generator model."""
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.stacks = stacks
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.skip_channels = skip_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.use_causal_conv = use_causal_conv
        self.upsample_conditional_features = upsample_conditional_features
        self.upsample_params = upsample_params
        self.initializer_seed = initializer_seed


class ParallelWaveGANDiscriminatorConfig(object):
    """Initialize ParallelWaveGAN Discriminator Config."""

    def __init__(
        self,
        out_channels=1,
        kernel_size=3,
        n_layers=10,
        conv_channels=64,
        use_bias=True,
        dilation_factor=1,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        initializer_seed=42,
        apply_sigmoid_at_last=False,
        **kwargs,
    ):
        "Init parameters for ParallelWaveGAN Discriminator model."
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.conv_channels = conv_channels
        self.use_bias = use_bias
        self.dilation_factor = dilation_factor
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.initializer_seed = initializer_seed
        self.apply_sigmoid_at_last = apply_sigmoid_at_last
