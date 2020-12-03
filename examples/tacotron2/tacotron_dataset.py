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
"""Tacotron Related Dataset modules."""

import itertools
import logging
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files


class CharactorMelDataset(AbstractDataset):
    """Tensorflow Charactor Mel dataset."""

    def __init__(
        self,
        dataset,
        root_dir,
        charactor_query="*-ids.npy",
        mel_query="*-norm-feats.npy",
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        mel_length_threshold=0,
        reduction_factor=1,
        mel_pad_value=0.0,
        char_pad_value=0,
        ga_pad_value=-1.0,
        g=0.2,
        use_fixed_shapes=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            reduction_factor (int): Reduction factor on Tacotron-2 paper.
            mel_pad_value (float): Padding value for mel-spectrogram.
            char_pad_value (int): Padding value for charactor.
            ga_pad_value (float): Padding value for guided attention.
            g (float): G value for guided attention.
            use_fixed_shapes (bool): Use fixed shape for mel targets or not.
            max_char_length (int): maximum charactor length if use_fixed_shapes=True.
            max_mel_length (int): maximum mel length if use_fixed_shapes=True

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
        char_lengths = [charactor_load_fn(f).shape[0] for f in charactor_files]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(mel_files) == len(charactor_files) == len(mel_lengths)
        ), f"Number of charactor, mel and duration files are different \
                ({len(mel_files)} vs {len(charactor_files)} vs {len(mel_lengths)})."

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.mel_files = mel_files
        self.charactor_files = charactor_files
        self.mel_load_fn = mel_load_fn
        self.charactor_load_fn = charactor_load_fn
        self.mel_lengths = mel_lengths
        self.char_lengths = char_lengths
        self.reduction_factor = reduction_factor
        self.mel_length_threshold = mel_length_threshold
        self.mel_pad_value = mel_pad_value
        self.char_pad_value = char_pad_value
        self.ga_pad_value = ga_pad_value
        self.g = g
        self.use_fixed_shapes = use_fixed_shapes
        self.max_char_length = np.max(char_lengths)

        if np.max(mel_lengths) % self.reduction_factor == 0:
            self.max_mel_length = np.max(mel_lengths)
        else:
            self.max_mel_length = (
                np.max(mel_lengths)
                + self.reduction_factor
                - np.max(mel_lengths) % self.reduction_factor
            )

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]

            items = {
                "utt_ids": utt_id,
                "mel_files": mel_file,
                "charactor_files": charactor_file,
            }

            yield items
    
    @tf.function
    def _load_data(self, items):
        mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        mel_length = len(mel)
        char_length = len(charactor)
        # padding mel to make its length is multiple of reduction factor.
        real_mel_length = mel_length
        remainder = mel_length % self.reduction_factor
        if remainder != 0:
            new_mel_length = mel_length + self.reduction_factor - remainder
            mel = tf.pad(
                mel,
                [[0, new_mel_length - mel_length], [0, 0]],
                constant_values=self.mel_pad_value,
            )
            mel_length = new_mel_length

        items = {
            "utt_ids": items["utt_ids"],
            "input_ids": charactor,
            "input_lengths": char_length,
            "speaker_ids": 0,
            "mel_gts": mel,
            "mel_lengths": mel_length,
            "real_mel_lengths": real_mel_length,
        }

        return items

    def _guided_attention(self, items):
        """Guided attention. Refer to page 3 on the paper (https://arxiv.org/abs/1710.08969)."""
        items = items.copy()
        mel_len = items["mel_lengths"] // self.reduction_factor
        char_len = items["input_lengths"]
        xv, yv = tf.meshgrid(tf.range(char_len), tf.range(mel_len), indexing="ij")
        f32_matrix = tf.cast(yv / mel_len - xv / char_len, tf.float32)
        items["g_attentions"] = 1.0 - tf.math.exp(
            -(f32_matrix ** 2) / (2 * self.g ** 2)
        )
        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
        drop_remainder=True
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        # load data
        datasets = datasets.map(
            lambda items: self._load_data(items),
            tf.data.experimental.AUTOTUNE
        )

        # calculate guided attention
        datasets = datasets.map(
            lambda items: self._guided_attention(items),
            tf.data.experimental.AUTOTUNE
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

        # define padding value.
        padding_values = {
            "utt_ids": " ",
            "input_ids": self.char_pad_value,
            "input_lengths": 0,
            "speaker_ids": 0,
            "mel_gts": self.mel_pad_value,
            "mel_lengths": 0,
            "real_mel_lengths": 0,
            "g_attentions": self.ga_pad_value,
        }

        # define padded shapes.
        padded_shapes = {
            "utt_ids": [],
            "input_ids": [None]
            if self.use_fixed_shapes is False
            else [self.max_char_length],
            "input_lengths": [],
            "speaker_ids": [],
            "mel_gts": [None, 80]
            if self.use_fixed_shapes is False
            else [self.max_mel_length, 80],
            "mel_lengths": [],
            "real_mel_lengths": [],
            "g_attentions": [None, None]
            if self.use_fixed_shapes is False
            else [self.max_char_length, self.max_mel_length // self.reduction_factor],
        }

        datasets = datasets.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder,
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mel_files": tf.string,
            "charactor_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorMelDataset"
