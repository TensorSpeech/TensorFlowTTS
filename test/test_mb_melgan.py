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

import tensorflow as tf

import logging
import os

import numpy as np
import pytest

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.models import TFPQMF, TFMelGANGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def make_multi_band_melgan_generator_args(**kwargs):
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
        subbands=4,
        tabs=62,
        cutoff_ratio=0.15,
        beta=9.0,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g",
    [
        {"subbands": 4, "upsample_scales": [2, 4, 8], "stacks": 4, "out_channels": 4},
        {"subbands": 4, "upsample_scales": [4, 4, 4], "stacks": 5, "out_channels": 4},
    ],
)
def test_multi_band_melgan(dict_g):
    args_g = make_multi_band_melgan_generator_args(**dict_g)
    args_g = MultiBandMelGANGeneratorConfig(**args_g)
    generator = TFMelGANGenerator(args_g, name="multi_band_melgan")
    generator._build()

    pqmf = TFPQMF(args_g, name="pqmf")

    fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
    fake_y = tf.random.uniform(shape=[1, 100 * 256, 1], dtype=tf.float32)
    y_hat_subbands = generator(fake_mels)

    y_hat = pqmf.synthesis(y_hat_subbands)
    y_subbands = pqmf.analysis(fake_y)

    assert np.shape(y_subbands) == np.shape(y_hat_subbands)
    assert np.shape(fake_y) == np.shape(y_hat)
