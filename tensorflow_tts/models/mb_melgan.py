# -*- coding: utf-8 -*-
# Copyright 2020 The Multi-band MelGAN Authors , Minh Nguyen (@dathudeptrai) and Tomoki Hayashi (@kan-bayashi)
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
# ============================================================================
#
# Compatible with https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/layers/pqmf.py.
"""Multi-band MelGAN Modules."""

import numpy as np
import tensorflow as tf
from scipy.signal import kaiser

from tensorflow_tts.models import TFMelGANGenerator


def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    # fix nan due to indeterminate form
    h_i[taps // 2] = np.cos(0) * cutoff_ratio

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class TFPQMF(tf.keras.layers.Layer):
    """PQMF module."""

    def __init__(self, config, **kwargs):
        """Initilize PQMF module.
        Args:
            config (class): MultiBandMelGANGeneratorConfig
        """
        super().__init__(**kwargs)
        subbands = config.subbands
        taps = config.taps
        cutoff_ratio = config.cutoff_ratio
        beta = config.beta

        # define filter coefficient
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # [subbands, 1, taps + 1] == [filter_width, in_channels, out_channels]
        analysis_filter = np.expand_dims(h_analysis, 1)
        analysis_filter = np.transpose(analysis_filter, (2, 1, 0))

        synthesis_filter = np.expand_dims(h_synthesis, 0)
        synthesis_filter = np.transpose(synthesis_filter, (2, 1, 0))

        # filter for downsampling & upsampling
        updown_filter = np.zeros((subbands, subbands, subbands), dtype=np.float32)
        for k in range(subbands):
            updown_filter[0, k, k] = 1.0

        self.subbands = subbands
        self.taps = taps
        self.analysis_filter = analysis_filter.astype(np.float32)
        self.synthesis_filter = synthesis_filter.astype(np.float32)
        self.updown_filter = updown_filter.astype(np.float32)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)],
    )
    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T, 1).
        Returns:
            Tensor: Output tensor (B, T // subbands, subbands).
        """
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])
        x = tf.nn.conv1d(x, self.analysis_filter, stride=1, padding="VALID")
        x = tf.nn.conv1d(x, self.updown_filter, stride=self.subbands, padding="VALID")
        return x

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)],
    )
    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T // subbands, subbands).
        Returns:
            Tensor: Output tensor (B, T, 1).
        """
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.subbands,
            strides=self.subbands,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.subbands,
                self.subbands,
            ),
        )
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])
        return tf.nn.conv1d(x, self.synthesis_filter, stride=1, padding="VALID")


class TFMBMelGANGenerator(TFMelGANGenerator):
    """Tensorflow MBMelGAN generator module."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.pqmf = TFPQMF(config=config, dtype=tf.float32, name="pqmf")

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference(self, mels):
        mb_audios = self.melgan(mels)
        return self.pqmf.synthesis(mb_audios)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference_tflite(self, mels):
        mb_audios = self.melgan(mels)
        return self.pqmf.synthesis(mb_audios)
