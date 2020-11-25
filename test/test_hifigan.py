# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team
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

import logging
import os

import pytest
import tensorflow as tf

from tensorflow_tts.configs import (
    HifiGANDiscriminatorConfig,
    HifiGANGeneratorConfig,
    MelGANDiscriminatorConfig,
)
from tensorflow_tts.models import (
    TFHifiGANGenerator,
    TFHifiGANMultiPeriodDiscriminator,
    TFMelGANMultiScaleDiscriminator,
)

from examples.hifigan.train_hifigan import TFHifiGANDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_hifigan_generator_args(**kwargs):
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return defaults


def make_hifigan_discriminator_args(**kwargs):
    defaults_multisperiod = dict(
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
    )
    defaults_multisperiod.update(kwargs)
    defaults_multiscale = dict(
        out_channels=1,
        scales=3,
        downsample_pooling="AveragePooling1D",
        downsample_pooling_params={"pool_size": 4, "strides": 2,},
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
    )
    defaults_multiscale.update(kwargs)
    return [defaults_multisperiod, defaults_multiscale]


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({"kernel_size": 3}, {}, {}),
        ({"filters": 1024}, {}, {}),
        ({"stack_kernel_size": [1, 2, 3]}, {}, {}),
        ({"stack_kernel_size": [3, 5, 7], "stacks": 3}, {}, {}),
        ({"upsample_scales": [4, 4, 4, 4]}, {}, {}),
        ({"upsample_scales": [8, 8, 2, 2]}, {}, {}),
        ({"filters": 1024, "upsample_scales": [8, 8, 2, 2]}, {}, {}),
    ],
)
def test_hifigan_trainable(dict_g, dict_d, dict_loss):
    batch_size = 4
    batch_length = 4096
    args_g = make_hifigan_generator_args(**dict_g)
    args_d_p, args_d_s = make_hifigan_discriminator_args(**dict_d)

    args_g = HifiGANGeneratorConfig(**args_g)
    args_d_p = HifiGANDiscriminatorConfig(**args_d_p)
    args_d_s = MelGANDiscriminatorConfig(**args_d_s)

    generator = TFHifiGANGenerator(args_g)

    discriminator_p = TFHifiGANMultiPeriodDiscriminator(args_d_p)
    discriminator_s = TFMelGANMultiScaleDiscriminator(args_d_s)
    discriminator = TFHifiGANDiscriminator(discriminator_p, discriminator_s)
