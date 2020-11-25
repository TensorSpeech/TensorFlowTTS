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

import itertools
import logging
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class CharactorDurationMelDataset(AbstractDataset):
    """Tensorflow Charactor Mel dataset."""

    def __init__(
        self,
        root_dir,
        charactor_query="*-ids.npy",
        mel_query="*-norm-feats.npy",
        duration_query="*-durations.npy",
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        duration_load_fn=np.load,
        mel_length_threshold=0,
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

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(mel_files) == len(charactor_files) == len(duration_files)
        ), f"Number of charactor, mel and duration files are different \
                ({len(mel_files)} vs {len(charactor_files)} vs {len(duration_files)})."

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.mel_files = mel_files
        self.charactor_files = charactor_files
        self.duration_files = duration_files
        self.mel_load_fn = mel_load_fn
        self.charactor_load_fn = charactor_load_fn
        self.duration_load_fn = duration_load_fn
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]
            duration_file = self.duration_files[i]

            items = {
                "utt_ids": utt_id,
                "mel_files": mel_file,
                "charactor_files": charactor_file,
                "duration_files": duration_file,
            }

            yield items

    @tf.function
    def _load_data(self, items):
        mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        duration = tf.numpy_function(np.load, [items["duration_files"]], tf.int32)

        items = {
            "utt_ids": items["utt_ids"],
            "input_ids": charactor,
            "speaker_ids": 0,
            "duration_gts": duration,
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

        # define padded_shapes
        padded_shapes = {
            "utt_ids": [],
            "input_ids": [None],
            "speaker_ids": [],
            "duration_gts": [None],
            "mel_gts": [None, None],
            "mel_lengths": [],
        }

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mel_files": tf.string,
            "charactor_files": tf.string,
            "duration_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDurationMelDataset"


class CharactorDataset(AbstractDataset):
    """Tensorflow Charactor dataset."""

    def __init__(
        self, root_dir, charactor_query="*-ids.npy", charactor_load_fn=np.load,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))

        # assert the number of files
        assert (
            len(charactor_files) != 0
        ), f"Not found any char or duration files in ${root_dir}."
        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.charactor_files = charactor_files
        self.charactor_load_fn = charactor_load_fn

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            charactor_file = self.charactor_files[i]
            charactor = self.charactor_load_fn(charactor_file)

            items = {"utt_ids": utt_id, "input_ids": charactor}

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

        # define padded shapes
        padded_shapes = {"utt_ids": [], "input_ids": [None]}

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=padded_shapes, drop_remainder=True
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {"utt_ids": tf.string, "input_ids": tf.int32}
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDataset"
