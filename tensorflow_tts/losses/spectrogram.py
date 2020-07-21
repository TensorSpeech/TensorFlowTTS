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
"""Spectrogram-based loss modules."""

import tensorflow as tf


class TFMelSpectrogram(tf.keras.layers.Layer):
    """Mel Spectrogram loss."""

    def __init__(
        self,
        n_mels=80,
        f_min=80.0,
        f_max=7600,
        frame_length=1024,
        frame_step=256,
        fft_length=1024,
        sample_rate=16000,
        **kwargs
    ):
        """Initialize."""
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            n_mels, fft_length // 2 + 1, sample_rate, f_min, f_max
        )

    def _calculate_log_mels_spectrogram(self, signals):
        """Calculate forward propagation.
        Args:
            signals (Tensor): signal (B, T).
        Returns:
            Tensor: Mel spectrogram (B, T', 80)
        """
        stfts = tf.signal.stft(
            signals,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
        )
        linear_spectrograms = tf.abs(stfts)
        mel_spectrograms = tf.tensordot(
            linear_spectrograms, self.linear_to_mel_weight_matrix, 1
        )
        mel_spectrograms.set_shape(
            linear_spectrograms.shape[:-1].concatenate(
                self.linear_to_mel_weight_matrix.shape[-1:]
            )
        )
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)  # prevent nan.
        return log_mel_spectrograms

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Mean absolute Error Spectrogram Loss.
        """
        y_mels = self._calculate_log_mels_spectrogram(y)
        x_mels = self._calculate_log_mels_spectrogram(x)
        return tf.reduce_mean(
            tf.abs(y_mels - x_mels), axis=list(range(1, len(x_mels.shape)))
        )
