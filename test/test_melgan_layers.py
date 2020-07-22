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

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_tts.models.melgan import (
    TFConvTranspose1d,
    TFReflectionPad1d,
    TFResidualStack,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize("padding_size", [(3), (5)])
def test_padding(padding_size):
    fake_input_1d = tf.random.normal(shape=[4, 8000, 256], dtype=tf.float32)
    out = TFReflectionPad1d(padding_size=padding_size)(fake_input_1d)
    assert np.array_equal(
        tf.keras.backend.int_shape(out), [4, 8000 + 2 * padding_size, 256]
    )


@pytest.mark.parametrize(
    "filters,kernel_size,strides,padding,is_weight_norm",
    [(512, 40, 8, "same", False), (768, 15, 8, "same", True)],
)
def test_convtranpose1d(filters, kernel_size, strides, padding, is_weight_norm):
    fake_input_1d = tf.random.normal(shape=[4, 8000, 256], dtype=tf.float32)
    conv1d_transpose = TFConvTranspose1d(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        is_weight_norm=is_weight_norm,
        initializer_seed=42,
    )
    out = conv1d_transpose(fake_input_1d)
    assert np.array_equal(tf.keras.backend.int_shape(out), [4, 8000 * strides, filters])


@pytest.mark.parametrize(
    "kernel_size,filters,dilation_rate,use_bias,nonlinear_activation,nonlinear_activation_params,is_weight_norm",
    [
        (3, 256, 1, True, "LeakyReLU", {"alpha": 0.3}, True),
        (3, 256, 3, True, "ReLU", {}, False),
    ],
)
def test_residualblock(
    kernel_size,
    filters,
    dilation_rate,
    use_bias,
    nonlinear_activation,
    nonlinear_activation_params,
    is_weight_norm,
):
    fake_input_1d = tf.random.normal(shape=[4, 8000, 256], dtype=tf.float32)
    residual_block = TFResidualStack(
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        nonlinear_activation=nonlinear_activation,
        nonlinear_activation_params=nonlinear_activation_params,
        is_weight_norm=is_weight_norm,
        initializer_seed=42,
    )
    out = residual_block(fake_input_1d)
    assert np.array_equal(tf.keras.backend.int_shape(out), [4, 8000, filters])
