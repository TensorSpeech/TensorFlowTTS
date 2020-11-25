# -*- coding: utf-8 -*-
# Copyright 2020 TensorflowTTS Team
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
"""HifiGAN Config object."""


class HifiGANGeneratorConfig(object):
    """Initialize HifiGAN Generator Config."""

    def __init__(
        self,
        out_channels=1,
        kernel_size=7,
        filters=128,
        use_bias=True,
        upsample_scales=[8, 8, 2, 2],
        stacks=3,
        stack_kernel_size=[3, 7, 11],
        stack_dilation_rate=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
        use_final_nolinear_activation=True,
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for HifiGAN Generator model."""
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.use_bias = use_bias
        self.upsample_scales = upsample_scales
        self.stacks = stacks
        self.stack_kernel_size = stack_kernel_size
        self.stack_dilation_rate = stack_dilation_rate
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.use_final_nolinear_activation = use_final_nolinear_activation
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed


class HifiGANDiscriminatorConfig(object):
    """Initialize HifiGAN Discriminator Config."""

    def __init__(
        self,
        out_channels=1,
        period_scales=[2, 3, 5, 7, 11],
        n_layers=5,
        kernel_size=5,
        strides=3,
        filters=8,
        filter_scales=4,
        max_filters=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for MelGAN Discriminator model."""
        self.out_channels = out_channels
        self.period_scales = period_scales
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.filter_scales = filter_scales
        self.max_filters = max_filters
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed
