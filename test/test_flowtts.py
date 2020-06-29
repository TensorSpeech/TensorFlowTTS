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

from typing import Dict
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_tts.models.flowtts import Squeeze2D
from tensorflow_tts.models.flowtts import Inv1x1Conv2DWithMask

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize("length, n_squeeze", [(256, 2), (256, 3), (300, 5), (500, 7)])
def test_squeeze_with_mask(length, n_squeeze):
    x = tf.random.uniform(shape=[4, length, 80], dtype=tf.float32)
    mask = tf.sequence_mask([128, 128, 200, 240], maxlen=length, dtype=tf.bool)
    mask = tf.expand_dims(mask, -1)
    squeeze_layer = Squeeze2D(with_zaux=False, name="squeeze")

    # forward
    z, z_mask = squeeze_layer.forward(x, n_squeeze=n_squeeze, mask=mask)

    # inverse
    x_inverse, _ = squeeze_layer.inverse(z, n_squeeze=n_squeeze, mask=z_mask)
    x = x * tf.cast(mask[:, :, :], dtype=x.dtype)
    x = x[:, : x.shape[1] // n_squeeze * n_squeeze, :]
    assert tf.less(tf.math.reduce_mean(x - x_inverse), 0.005)


@pytest.mark.parametrize("max_length", [(256), (512), (240),])
def test_inv_conv1x1_with_mask(max_length):
    x = tf.random.uniform(shape=[4, max_length, 80], dtype=tf.float32)
    mask = tf.sequence_mask([128, 128, 200, 240], maxlen=max_length, dtype=tf.bool)
    mask = tf.expand_dims(mask, axis=-1)  # [B, T, 1]
    inv_conv1x1 = Inv1x1Conv2DWithMask(name="inv_conv1x1")
    inv_conv1x1(x)

    z, log_det_jacobian = inv_conv1x1.forward(x, mask=mask)
    x_inverse, inverse_log_det_jacobian = inv_conv1x1.inverse(z, mask=mask)

    assert tf.less(tf.math.reduce_mean(x * tf.cast(mask, x.dtype) - x_inverse), 0.005)
