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
"""Dataset modules."""

import os
import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


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
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        duration_load_fn=np.load,
        f0_load_fn=np.load,
        energy_load_fn=np.load,
        mel_length_threshold=0,
        speakers_map=None
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            duration_query (str): Query to find duration files in root_dir.
            f0_query (str): Query to find f0 files in root_dir.
            energy_query (str): Query to find energy files in root_dir.
            f0_stat (str): str path of f0_stat.
            energy_stat (str): str path of energy_stat.
            charactor_load_fn (func): Function to load charactor file.
            mel_load_fn (func): Function to load feature file.
            duration_load_fn (func): Function to load duration file.
            f0_load_fn (func): Function to load f0 file.
            energy_load_fn (func): Function to load energy file.
            mel_length_threshold (int): Threshold to remove short feature files.
            speakers_map (dict): Speakers map generated in dataset preprocessing

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        duration_files = sorted(find_files(root_dir, duration_query))
        f0_files = sorted(find_files(root_dir, f0_query))
        energy_files = sorted(find_files(root_dir, energy_query))

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(mel_files)
            == len(charactor_files)
            == len(duration_files)
            == len(f0_files)
            == len(energy_files)
        ), f"Number of charactor, mel, duration, f0 and energy files are different"

        assert speakers_map != None, f"No speakers map found. Did you set --dataset_mapping?"

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
        self.mel_length_threshold = mel_length_threshold
        self.speakers_map = speakers_map
        self.speakers = [self.speakers_map[i.split("_")[0]] for i in self.utt_ids]
        print("Speaker: utt_id", list(zip(self.speakers, self.utt_ids)))
        self.f0_stat = np.load(f0_stat)
        self.energy_stat = np.load(energy_stat)

    def get_args(self):
        return [self.utt_ids]

    def _norm_mean_std(self, x, mean, std):
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x

    def _norm_mean_std_tf(self, x, mean, std):
        x = tf.numpy_function(self._norm_mean_std, [x, mean, std], tf.float32)
        return x

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]
            duration_file = self.duration_files[i]
            f0_file = self.f0_files[i]
            energy_file = self.energy_files[i]
            speaker_id = self.speakers[i]

            items = {
                "utt_ids": utt_id,
                "mel_files": mel_file,
                "charactor_files": charactor_file,
                "duration_files": duration_file,
                "f0_files": f0_file,
                "energy_files": energy_file,
                "speaker_ids": speaker_id,
            }

            yield items

    @tf.function
    def _load_data(self, items):
        mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        duration = tf.numpy_function(np.load, [items["duration_files"]], tf.int32)
        f0 = tf.numpy_function(np.load, [items["f0_files"]], tf.float32)
        energy = tf.numpy_function(np.load, [items["energy_files"]], tf.float32)

        f0 = self._norm_mean_std_tf(f0, self.f0_stat[0], self.f0_stat[1])
        energy = self._norm_mean_std_tf(
            energy, self.energy_stat[0], self.energy_stat[1]
        )

        # calculate charactor f0/energy
        f0 = tf_average_by_duration(f0, duration)
        energy = tf_average_by_duration(energy, duration)

        items = {
            "utt_ids": items["utt_ids"],
            "input_ids": charactor,
            "speaker_ids": items["speaker_ids"],
            "duration_gts": duration,
            "f0_gts": f0,
            "energy_gts": energy,
            "mel_gts": mel,
            "mel_lengths": len(mel),
        }

        return items

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

        # load data
        datasets = datasets.map(
            lambda items: self._load_data(items), tf.data.experimental.AUTOTUNE
        )

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {
            "utt_ids": [],
            "input_ids": [None],
            "speaker_ids": [],
            "duration_gts": [None],
            "f0_gts": [None],
            "energy_gts": [None],
            "mel_gts": [None, None],
            "mel_lengths": [],
        }

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=padded_shapes, drop_remainder=True
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mel_files": tf.string,
            "charactor_files": tf.string,
            "duration_files": tf.string,
            "f0_files": tf.string,
            "energy_files": tf.string,
            "speaker_ids": tf.int32,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDurationF0EnergyMelDataset"
