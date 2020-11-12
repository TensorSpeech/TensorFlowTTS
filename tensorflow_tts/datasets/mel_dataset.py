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

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class MelDataset(AbstractDataset):
    """Tensorflow compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-raw-feats.h5",
        mel_load_fn=np.load,
        mel_length_threshold=0,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.

        """
        # find all of mel files.
        mel_files = sorted(find_files(root_dir, mel_query))
        mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        if ".npy" in mel_query:
            suffix = mel_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in mel_files]

        # set global params
        self.utt_ids = utt_ids
        self.mel_files = mel_files
        self.mel_lengths = mel_lengths
        self.mel_load_fn = mel_load_fn
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            mel = self.mel_load_fn(mel_file)
            mel_length = self.mel_lengths[i]

            items = {"utt_ids": utt_id, "mels": mel, "mel_lengths": mel_length}

            yield items

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mels": tf.float32,
            "mel_lengths": tf.int32,
        }
        return output_types

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
            "mels": [None, 80],
            "mel_lengths": [],
        }

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "MelDataset"
