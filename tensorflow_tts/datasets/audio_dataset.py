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
"""Audio modules."""

import logging
import os
import numpy as np

import tensorflow as tf

from tensorflow_tts.utils import find_files

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset


class AudioDataset(AbstractDataset):
    """Tensorflow compatible audio dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        audio_load_fn=np.load,
        audio_length_threshold=None,
        return_utt_id=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of mel files.
        audio_files = sorted(find_files(root_dir, audio_query))
        audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]

        # filter by threshold
        if audio_length_threshold is not None:
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any mel files in ${root_dir}."

        if ".npy" in audio_query:
            suffix = audio_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in audio_files]

        # set global params
        self.utt_ids = utt_ids
        self.audio_files = audio_files
        self.audio_lengths = audio_lengths
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            audio_file = self.audio_files[i]
            audio = self.audio_load_fn(audio_file)
            audio_length = self.audio_lengths[i]
            if self.return_utt_id:
                items = utt_id, audio, audio_length
            else:
                items = audio, audio_length
            yield items

    def get_output_dtypes(self):
        output_types = (tf.float32, tf.int32)
        if self.return_utt_id:
            output_types = (tf.dtypes.string, *output_types)
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

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded_shapes.
        padded_shapes = ([None,], [])
        if self.return_utt_id:
            padded_shapes = ([], *padded_shapes)

        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "AudioDataset"
