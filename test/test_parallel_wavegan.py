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

import logging
import os

import pytest
import tensorflow as tf

from tensorflow_tts.configs import (
    ParallelWaveGANGeneratorConfig,
    ParallelWaveGANDiscriminatorConfig,
)
from tensorflow_tts.models import (
    TFParallelWaveGANGenerator,
    TFParallelWaveGANDiscriminator,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_pwgan_generator_args(**kwargs):
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return defaults


def make_pwgan_discriminator_args(**kwargs):
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d",
    [
        ({}, {}),
        (
            {"kernel_size": 3, "aux_context_window": 5, "residual_channels": 128},
            {"dilation_factor": 2},
        ),
        ({"stacks": 4, "n_layers": 40}, {"conv_channels": 128}),
    ],
)
def test_melgan_trainable(dict_g, dict_d):
    random_c = tf.random.uniform(shape=[4, 32, 80], dtype=tf.float32)

    args_g = make_pwgan_generator_args(**dict_g)
    args_d = make_pwgan_discriminator_args(**dict_d)

    args_g = ParallelWaveGANGeneratorConfig(**args_g)
    args_d = ParallelWaveGANDiscriminatorConfig(**args_d)

    generator = TFParallelWaveGANGenerator(args_g)
    generator._build()
    discriminator = TFParallelWaveGANDiscriminator(args_d)
    discriminator._build()

    generated_audios = generator(random_c, training=True)
    discriminator(generated_audios)

    generator.summary()
    discriminator.summary()
