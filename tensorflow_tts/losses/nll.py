# -*- coding: utf-8 -*-
# Copyright 2020 Mokke Meguru (@MokkeMeguru)
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
from TFGENZOO.flows.utils import gaussian_likelihood


def nll(z: tf.Tensor, mask: tf.Tensor):
    """negative log likelihood for z

    Args:
       z (tf.Tensor): base latent variable [B, T, C]
    Returns:
       tf.Tensor: nll [B, T, C]
    """
    ll = gaussian_likelihood(tf.zeros(tf.shape(z)), tf.zeros(tf.shape(z)), z)
    return ll * tf.cast(mask, z.dtype)
