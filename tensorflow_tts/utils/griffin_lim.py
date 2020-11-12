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
"""Griffin-Lim phase reconstruction algorithm from mel spectrogram."""

import os

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def griffin_lim_lb(
    mel_spec, stats_path, dataset_config, n_iter=32, output_dir=None, wav_name="lb"
):
    """Generate wave from mel spectrogram with Griffin-Lim algorithm using Librosa.
    Args:
        mel_spec (ndarray): array representing the mel spectrogram.
        stats_path (str): path to the `stats.npy` file containing norm statistics.
        dataset_config (Dict): dataset configuration parameters.
        n_iter (int): number of iterations for GL.
        output_dir (str): output directory where audio file will be saved.
        wav_name (str): name of the output file.
    Returns:
        gl_lb (ndarray): generated wave.
    """
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.load(stats_path)

    mel_spec = np.power(10.0, scaler.inverse_transform(mel_spec)).T
    mel_basis = librosa.filters.mel(
        dataset_config["sampling_rate"],
        n_fft=dataset_config["fft_size"],
        n_mels=dataset_config["num_mels"],
        fmin=dataset_config["fmin"],
        fmax=dataset_config["fmax"],
    )
    mel_to_linear = np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec))
    gl_lb = librosa.griffinlim(
        mel_to_linear,
        n_iter=n_iter,
        hop_length=dataset_config["hop_size"],
        win_length=dataset_config["win_length"] or dataset_config["fft_size"],
    )
    if output_dir:
        output_path = os.path.join(output_dir, f"{wav_name}.wav")
        sf.write(output_path, gl_lb, dataset_config["sampling_rate"], "PCM_16")
    return gl_lb


class TFGriffinLim(tf.keras.layers.Layer):
    """Griffin-Lim algorithm for phase reconstruction from mel spectrogram magnitude."""

    def __init__(self, stats_path, dataset_config, normalized: bool = True):
        """Init GL params.
        Args:
            stats_path (str): path to the `stats.npy` file containing norm statistics.
            dataset_config (Dict): dataset configuration parameters.
        """
        super().__init__()
        self.normalized = normalized
        if normalized:
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = np.load(stats_path)
            self.scaler = scaler
        self.ds_config = dataset_config
        self.mel_basis = librosa.filters.mel(
            self.ds_config["sampling_rate"],
            n_fft=self.ds_config["fft_size"],
            n_mels=self.ds_config["num_mels"],
            fmin=self.ds_config["fmin"],
            fmax=self.ds_config["fmax"],
        )  # [num_mels, fft_size // 2 + 1]

    def save_wav(self, gl_tf, output_dir, wav_name):
        """Generate WAV file and save it.
        Args:
            gl_tf (tf.Tensor): reconstructed signal from GL algorithm.
            output_dir (str): output directory where audio file will be saved.
            wav_name (str): name of the output file.
        """
        encode_fn = lambda x: tf.audio.encode_wav(x, self.ds_config["sampling_rate"])
        gl_tf = tf.expand_dims(gl_tf, -1)
        if not isinstance(wav_name, list):
            wav_name = [wav_name]

        if len(gl_tf.shape) > 2:
            bs, *_ = gl_tf.shape
            assert bs == len(wav_name), "Batch and 'wav_name' have different size."
            tf_wav = tf.map_fn(encode_fn, gl_tf, dtype=tf.string)
            for idx in tf.range(bs):
                output_path = os.path.join(output_dir, f"{wav_name[idx]}.wav")
                tf.io.write_file(output_path, tf_wav[idx])
        else:
            tf_wav = encode_fn(gl_tf)
            tf.io.write_file(os.path.join(output_dir, f"{wav_name[0]}.wav"), tf_wav)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        ]
    )
    def call(self, mel_spec, n_iter=32):
        """Apply GL algorithm to batched mel spectrograms.
        Args:
            mel_spec (tf.Tensor): normalized mel spectrogram.
            n_iter (int): number of iterations to run GL algorithm.
        Returns:
            (tf.Tensor): reconstructed signal from GL algorithm.
        """
        # de-normalize mel spectogram
        if self.normalized:
            mel_spec = tf.math.pow(
                10.0, mel_spec * self.scaler.scale_ + self.scaler.mean_
            )
        else:
            mel_spec = tf.math.pow(
                10.0, mel_spec
            )  # TODO @dathudeptrai check if its ok without it wavs were too quiet
        inverse_mel = tf.linalg.pinv(self.mel_basis)

        # [:, num_mels] @ [fft_size // 2 + 1, num_mels].T
        mel_to_linear = tf.linalg.matmul(mel_spec, inverse_mel, transpose_b=True)
        mel_to_linear = tf.cast(tf.math.maximum(1e-10, mel_to_linear), tf.complex64)

        init_phase = tf.cast(
            tf.random.uniform(tf.shape(mel_to_linear), maxval=1), tf.complex64
        )
        phase = tf.math.exp(2j * np.pi * init_phase)
        for _ in tf.range(n_iter):
            inverse = tf.signal.inverse_stft(
                mel_to_linear * phase,
                frame_length=self.ds_config["win_length"] or self.ds_config["fft_size"],
                frame_step=self.ds_config["hop_size"],
                fft_length=self.ds_config["fft_size"],
                window_fn=tf.signal.inverse_stft_window_fn(self.ds_config["hop_size"]),
            )
            phase = tf.signal.stft(
                inverse,
                self.ds_config["win_length"] or self.ds_config["fft_size"],
                self.ds_config["hop_size"],
                self.ds_config["fft_size"],
            )
            phase /= tf.cast(tf.maximum(1e-10, tf.abs(phase)), tf.complex64)

        return tf.signal.inverse_stft(
            mel_to_linear * phase,
            frame_length=self.ds_config["win_length"] or self.ds_config["fft_size"],
            frame_step=self.ds_config["hop_size"],
            fft_length=self.ds_config["fft_size"],
            window_fn=tf.signal.inverse_stft_window_fn(self.ds_config["hop_size"]),
        )
