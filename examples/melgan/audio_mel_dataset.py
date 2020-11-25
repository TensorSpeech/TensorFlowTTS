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


class AudioMelDataset(AbstractDataset):
    """Tensorflow Audio Mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        mel_query="*-raw-feats.npy",
        audio_load_fn=np.load,
        mel_load_fn=np.load,
        audio_length_threshold=0,
        mel_length_threshold=0,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of audio and mel files.
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(
            mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        if ".npy" in audio_query:
            suffix = audio_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in audio_files]

        # set global params
        self.utt_ids = utt_ids
        self.audio_files = audio_files
        self.mel_files = mel_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.audio_length_threshold = audio_length_threshold
        self.mel_length_threshold = mel_length_threshold

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            audio_file = self.audio_files[i]
            mel_file = self.mel_files[i]

            items = {
                "utt_ids": utt_id,
                "audio_files": audio_file,
                "mel_files": mel_file,
            }

            yield items

    @tf.function
    def _load_data(self, items):
        audio = tf.numpy_function(np.load, [items["audio_files"]], tf.float32)
        mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)

        items = {
            "utt_ids": items["utt_ids"],
            "audios": audio,
            "mels": mel,
            "mel_lengths": len(mel),
            "audio_lengths": len(audio),
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

        # load dataset
        datasets = datasets.map(
            lambda items: self._load_data(items), tf.data.experimental.AUTOTUNE
        )

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )
        datasets = datasets.filter(
            lambda x: x["audio_lengths"] > self.audio_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            datasets = datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        # define padded shapes
        padded_shapes = {
            "utt_ids": [],
            "audios": [None],
            "mels": [None, 80],
            "mel_lengths": [],
            "audio_lengths": [],
        }

        # define padded values
        padding_values = {
            "utt_ids": "",
            "audios": 0.0,
            "mels": 0.0,
            "mel_lengths": 0,
            "audio_lengths": 0,
        }

        datasets = datasets.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "audio_files": tf.string,
            "mel_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "AudioMelDataset"
