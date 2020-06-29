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
"""Convert Melspectrogram to wav by GL algorithm."""
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
import librosa


class TFGriffinLim(tf.keras.layers.Layer):
    """GL algorithm."""

    def __init__(self, config, **kwargs):
        """Init GL params."""
        super().__init__(**kwargs)
        self.config = config
        self._n_iters = 60

    def setup_stats(self, stats):
        """Setup mel mean/var."""
        scaler = StandardScaler()
        scaler.mean_ = stats[0]
        scaler.scale_ = stats[1]
        self._scaler = scaler

    def _de_normalization(self, mel_spectrogram):
        """Convert norm-mels to raw-mels."""
        return self._scaler.inverse_transform(mel_spectrogram)

    @tf.function
    def _build_mel_basis(self):
        """Build mel basis."""
        return librosa.filters.mel(self.config["sampling_rate"],
                                   self.config["fft_size"],
                                   n_mels=self.config["num_mels"],
                                   fmin=self.config["fmin"],
                                   fmax=self.config["fmax"])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def _mel_to_linear(self, mel_spectrogram):
        """Convert mel to linear spectrogram."""
        _inv_mel_basis = tf.linalg.pinv(self._build_mel_basis())
        return tf.math.maximum(1e-10, tf.matmul(_inv_mel_basis, tf.transpose(mel_spectrogram, (1, 0))))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.complex64)])
    def _invert_spectrogram(self, spectrogram):
        """Invert Spectrogram."""
        spectrogram = tf.expand_dims(spectrogram, 0)
        inversed = tf.signal.inverse_stft(
            spectrogram,
            self.config["fft_size"] if self.config["win_length"] is None else self.config["win_length"],
            self.config["hop_size"],
            self.config["fft_size"]
        )
        return tf.squeeze(inversed, 0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def run_convert(self, mel_spectrogram):
        """Run convert mel-spectrogram to wav-form."""
        spectrogram = self._mel_to_linear(tf.pow(10.0, mel_spectrogram))
        spectrogram = tf.transpose(spectrogram, (1, 0))
        spectrogram = tf.cast(spectrogram, dtype=tf.complex64)
        best = tf.identity(spectrogram)

        for _ in tf.range(self._n_iters):
            best = self._invert_spectrogram(spectrogram)
            estimate = tf.signal.stft(
                best,
                self.config["fft_size"] if self.config["win_length"] is None else self.config["win_length"],
                self.config["hop_size"],
                self.config["fft_size"]
            )
            phase = estimate / tf.cast(tf.maximum(1e-10, tf.abs(estimate)), tf.complex64)
            best = spectrogram * phase

        y = tf.math.real(self._invert_spectrogram(best))
        return y

    def call(self, mel_spectrogram):
        """Call logic."""
        mel_spectrogram = self._de_normalization(mel_spectrogram)
        return self.run_convert(mel_spectrogram)
