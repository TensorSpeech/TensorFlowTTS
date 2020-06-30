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
"""Dataset modules."""

import logging
import os
import random
import itertools
import numpy as np

import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset

from tensorflow_tts.utils import find_files
from tensorflow_tts.utils import remove_outlier


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


@tf.function(
    input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.int32)]
)
def tf_average_by_duration(x, durs):
    outs = tf.numpy_function(average_by_duration, [x, durs], tf.float32)
    return outs


class CharactorDurationF0EnergyMelDataset(AbstractDataset):
    """Tensorflow Charactor Duration F0 Energy Mel dataset."""

    def __init__(
        self,
        root_dir,
        charactor_query="*-ids.npy",
        mel_query="*-norm-feats.npy",
        duration_query="*-durations.npy",
        f0_query="*-raw-f0.npy",
        energy_query="*-raw-energy.npy",
        f0_stat="./dump/stats_f0.npy",
        energy_stat="./dump/stats_energy.npy",
        max_f0_embeddings=256,
        max_energy_embeddings=256,
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        duration_load_fn=np.load,
        f0_load_fn=np.load,
        energy_load_fn=np.load,
        mel_length_threshold=None,
        return_utt_id=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            duration_query (str): Query to find duration files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            mel_load_fn (func): Function to load feature file.
            duration_load_fn (func): Function to load duration file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        duration_files = sorted(find_files(root_dir, duration_query))
        f0_files = sorted(find_files(root_dir, f0_query))
        energy_files = sorted(find_files(root_dir, energy_query))
        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]

            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]
            charactor_files = [charactor_files[idx] for idx in idxs]
            duration_files = [duration_files[idx] for idx in idxs]
            mel_lengths = [mel_lengths[idx] for idx in idxs]
            f0_files = [f0_files[idx] for idx in idxs]
            energy_files = [energy_files[idx] for idx in idxs]

            # bucket sequence length trick, sort based-on mel-length.
            idx_sort = np.argsort(mel_lengths)

            # sort
            mel_files = np.array(mel_files)[idx_sort]
            charactor_files = np.array(charactor_files)[idx_sort]
            duration_files = np.array(duration_files)[idx_sort]
            f0_files = np.array(f0_files)[idx_sort]
            energy_files = np.array(energy_files)[idx_sort]
            mel_lengths = np.array(mel_lengths)[idx_sort]

            # group
            idx_lengths = [
                [idx, length]
                for idx, length in zip(np.arange(len(mel_lengths)), mel_lengths)
            ]
            groups = [
                list(g) for _, g in itertools.groupby(idx_lengths, lambda a: a[1])
            ]

            # group shuffle
            random.shuffle(groups)

            # get idxs affter group shuffle
            idxs = []
            for group in groups:
                for idx, _ in group:
                    idxs.append(idx)

            # re-arange dataset
            mel_files = np.array(mel_files)[idxs]
            charactor_files = np.array(charactor_files)[idxs]
            duration_files = np.array(duration_files)[idxs]
            f0_files = np.array(f0_files)[idxs]
            energy_files = np.array(energy_files)[idxs]
            mel_lengths = np.array(mel_lengths)[idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(mel_files)
            == len(charactor_files)
            == len(duration_files)
            == len(f0_files)
            == len(energy_files)
        ), f"Number of charactor, mel, duration, f0 and energy files are different"

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.mel_files = mel_files
        self.charactor_files = charactor_files
        self.duration_files = duration_files
        self.f0_files = f0_files
        self.energy_files = energy_files
        self.mel_load_fn = mel_load_fn
        self.charactor_load_fn = charactor_load_fn
        self.duration_load_fn = duration_load_fn
        self.f0_load_fn = f0_load_fn
        self.energy_load_fn = energy_load_fn
        self.return_utt_id = return_utt_id

        self.f0_stat = np.load(f0_stat)
        self.energy_stat = np.load(energy_stat)
        self.max_f0_embeddings = max_f0_embeddings
        self.max_energy_embeddings = max_energy_embeddings

    def get_args(self):
        return [self.utt_ids]

    def _norm_mean_std(self, x, mean, std):
        x = remove_outlier(x)
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]
            duration_file = self.duration_files[i]
            f0_file = self.f0_files[i]
            energy_file = self.energy_files[i]
            mel = self.mel_load_fn(mel_file)
            charactor = self.charactor_load_fn(charactor_file)
            duration = self.duration_load_fn(duration_file)
            f0 = self.f0_load_fn(f0_file)
            energy = self.energy_load_fn(energy_file)

            f0 = self._norm_mean_std(f0, self.f0_stat[0], self.f0_stat[1])
            energy = self._norm_mean_std(
                energy, self.energy_stat[0], self.energy_stat[1]
            )

            # calculate charactor f0/energy
            f0 = tf_average_by_duration(f0, duration)
            energy = tf_average_by_duration(energy, duration)

            if self.return_utt_id:
                items = utt_id, charactor, duration, f0, energy, mel
            else:
                items = charactor, duration, f0, energy, mel
            yield items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=([None], [None], [None], [None], [None, None])
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = (tf.int32, tf.int32, tf.float32, tf.float32, tf.float32)
        if self.return_utt_id:
            output_types = (tf.dtypes.string, *output_types)
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDurationF0EnergyMelDataset"
