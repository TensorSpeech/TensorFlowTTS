# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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

from tensorflow_tts.configs import MelGANDiscriminatorConfig, MelGANGeneratorConfig
from tensorflow_tts.models import TFMelGANGenerator, TFMelGANMultiScaleDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_melgan_generator_args(**kwargs):
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return defaults


def make_melgan_discriminator_args(**kwargs):
    defaults = dict(
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
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss",
    [
        ({}, {}, {}),
        ({"kernel_size": 3}, {}, {}),
        ({"filters": 1024}, {}, {}),
        ({"stack_kernel_size": 5}, {}, {}),
        ({"stack_kernel_size": 5, "stacks": 2}, {}, {}),
        ({"upsample_scales": [4, 4, 4, 4]}, {}, {}),
        ({"upsample_scales": [8, 8, 2, 2]}, {}, {}),
        ({"filters": 1024, "upsample_scales": [8, 8, 2, 2]}, {}, {}),
    ],
)
def test_melgan_trainable(dict_g, dict_d, dict_loss):
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_melgan_discriminator_args(**dict_d)

    args_g = MelGANGeneratorConfig(**args_g)
    args_d = MelGANDiscriminatorConfig(**args_d)

    generator = TFMelGANGenerator(args_g)
    discriminator = TFMelGANMultiScaleDiscriminator(args_d)
