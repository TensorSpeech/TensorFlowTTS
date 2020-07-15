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
"""Tacotron 2 related dataset modules."""

import glob
import os

import numpy as np
import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.processor.ljspeech import symbols as ljs_symbols

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CharactorMelDataset(AbstractDataset):
    """Dataset class to find, load and transform character and mel spectrogram data for Tacotron2."""

    def __init__(
        self,
        dataset_dir,
        dataset="ljspeech",
        dataset_format="npy",
        use_norm=False,
        stats_path=None,
        return_guided_attention=True,
        reduction_factor=1,
        mel_pad_value=0.0,
        char_pad_value=0,
        ga_pad_value=-1.0,
        g=0.2,
        n_mels=80,
        use_fixed_shapes=False,
        mel_len_threshold=None,
    ):
        """Initialize dataset.
        Args:
            dataset_dir (str): Data directory where files are located.
            dataset_format (str): Dataset format the dataset has been saved.
            use_norm (bool): Whether or not to standardize data.
            stats_path (str): Path to the statistics file with mean and std values for standardization.
            return_guided_attention (bool): Whether or not guided attention matrix should be computed.
            reduction_factor (int): Reduction factor on Tacotron-2 paper.
            mel_pad_value (int): Padding value for mel spectrogram.
            char_pad_value (int): Padding value for characters.
            ga_pad_value (float): Padding value for guided attention.
            g (float): G value for guided attention.
            n_mels (int): Number of mel features.
            mel_query (str): Query to find feature files in root_dir.
            use_fixed_shapes (bool): Whether or not to use fixed shapes.
            mel_length_threshold (int): Threshold to remove short feature files.
        """
        # find all character and mel files
        if dataset_format == "npy":
            self.load_fn = np.load
            char_files_path = os.path.join(dataset_dir, "ids", "*-ids.npy")
            mel_files_path = os.path.join(dataset_dir, "raw-feats", "*-raw-feats.npy")

            char_files = sorted(glob.glob(char_files_path))
            mel_files = sorted(glob.glob(mel_files_path))

            utt_ids = [os.path.basename(x)[:10] for x in char_files]

            # check directory is populated
            assert (
                len(char_files) != 0 and len(mel_files) != 0
            ), f"No files found in '{dataset_dir}'."
            # check the char_files and mel_files have same ID
            assert utt_ids == [
                os.path.basename(x)[:10] for x in mel_files
            ], "File name mismatch between 'char_files' and 'mel_files'."

            features = zip(utt_ids, char_files, mel_files)
        else:
            raise ValueError("'dataset_format' only supports 'npy'.")

        if use_norm:
            if stats_path:
                self.mel_mean, self.mel_scale = np.load(stats_path)
            else:
                raise ValueError(
                    "'stats_path' needs to be provided if 'use_norm' is enabled."
                )

        self.dataset = dataset
        self.max_char_length = None  # 190 for LJSpeech
        self.max_mel_length = None  # 870 for LJSpeech
        self.use_fixed_shapes = use_fixed_shapes
        if use_fixed_shapes:
            char_len = [self.load_fn(f).shape[0] for f in char_files]
            self.max_char_length = np.max(char_len) + 1  # +1 for "eos"

            mel_len = [self.load_fn(f).shape[0] for f in mel_files]
            remainder = np.max(mel_len) % reduction_factor
            if remainder == 0:
                self.max_mel_length = np.max(mel_len)
            else:
                self.max_mel_length = np.max(mel_len) + reduction_factor - remainder

        # set global params
        self.features = list(features)
        self.use_norm = use_norm
        self.mel_len_threshold = mel_len_threshold
        self.return_guided_attention = return_guided_attention
        self.reduction_factor = reduction_factor
        self.g = g
        self.pad_value = {
            "mel_gts": mel_pad_value,
            "input_ids": char_pad_value,
            "g_attentions": ga_pad_value,
        }
        self.n_mels = n_mels

    def get_args(self):
        return [self.features]

    def get_output_dtypes(self):
        return {
            "input_ids": tf.int32,
            "input_lengths": tf.int32,
            "mel_gts": tf.float32,
            "mel_lengths": tf.int32,
            "speaker_ids": tf.int32,
            "utt_ids": tf.string,
        }

    def get_len_dataset(self):
        return len(self.features)

    def generator(self, features):
        for utt_id, char_file, mel_file in features:
            char = self.load_fn(char_file)
            mel = self.load_fn(mel_file)
            items = {
                "input_ids": char,
                "input_lengths": char.shape[0],
                "mel_gts": mel,
                "mel_lengths": mel.shape[0],
                "speaker_ids": 0,
                "utt_ids": utt_id,
            }
            yield items

    @tf.function
    def add_eos_and_pad(self, items, eos_token):
        """Adds "eos" token and pads if necessary wrt reduction factor."""
        items = items.copy()
        # padding mel to make its length multiple of reduction factor
        remainder = items["mel_lengths"] % self.reduction_factor
        if remainder != 0:
            new_mel_len = items["mel_lengths"] + self.reduction_factor - remainder
            items["mel_gts"] = tf.pad(
                items["mel_gts"],
                [[0, new_mel_len - items["mel_lengths"]], [0, 0]],
                constant_values=self.pad_value["mel_gts"],
            )
            items["mel_lengths"] = new_mel_len
        # add "eos" token for character at the end of the sentence
        items["input_ids"] = tf.concat([items["input_ids"], [eos_token]], -1)
        items["input_lengths"] += 1
        return items

    @tf.function
    def mel_norm(self, items):
        """Standardize mel features."""
        items = items.copy()
        items["mel_gts"] = (items["mel_gts"] - self.mel_mean) / self.mel_scale
        return items

    @tf.function
    def guided_attention(self, items):
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
        batch_size,
        allow_cache=False,
        is_shuffle=False,
        reshuffle_each_iteration=True,
        training=False,
    ):
        """Create tf.data.Dataset function and apply requested transformations.
        Args:
            batch_size (int): Size of batch used per iteration.
            allow_cache (bool): Whether the processed dataset should be stored in memory or not.
            shuffle (bool): Whether dataset should be shuffled.
            reshuffle_each_iteration (bool): Whether a new shuffle should be applied each epoch.
            training (bool): Whether the dataset is for training or validation.
        Returns:
            ds (tf.data.Dataset): Dataset containing all transformations.
        """
        ds = tf.data.Dataset.from_generator(
            self.generator, output_types=self.get_output_dtypes(), args=self.get_args()
        )

        # drop dataset elements that are shorter than "mel_len_threshold"
        if self.mel_len_threshold:
            ds = ds.filter(lambda x: x["mel_lengths"] > self.mel_len_threshold)

        # standardize features if requested
        if self.use_norm:
            ds = ds.map(self.mel_norm, num_parallel_calls=AUTOTUNE)

        # add "eos" and pad mel spectrogram given reduction factor
        dataset_symbols = {
            "ljspeech": ljs_symbols,
        }
        eos_token = len(dataset_symbols[self.dataset]) - 1
        ds = ds.map(
            lambda x: self.add_eos_and_pad(x, tf.constant(eos_token)),
            num_parallel_calls=AUTOTUNE,
        )

        # return guided attention if requested
        if self.return_guided_attention:
            ds = ds.map(self.guided_attention, num_parallel_calls=AUTOTUNE)

        if allow_cache:
            ds = ds.cache()
        if is_shuffle and training:
            ds = ds.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define default padding value for each element in dataset
        padding_values = {
            "input_ids": self.pad_value["input_ids"],
            "input_lengths": 0,
            "mel_gts": self.pad_value["mel_gts"],
            "mel_lengths": 0,
            "speaker_ids": 0,
            "utt_ids": " ",
        }
        # define default padded shapes for each element in dataset
        padded_shapes = {
            "input_ids": [None],
            "input_lengths": [],
            "mel_gts": [None, self.n_mels],
            "mel_lengths": [],
            "speaker_ids": [],
            "utt_ids": [],
        }

        if self.return_guided_attention:
            padding_values["g_attentions"] = self.pad_value["g_attentions"]
            padded_shapes["g_attentions"] = (
                [self.max_char_length, self.max_mel_length // self.reduction_factor]
                if self.use_fixed_shapes
                else [None, None]
            )

        if self.use_fixed_shapes:
            padded_shapes["input_ids"] = [self.max_char_length]
            padded_shapes["mel_gts"] = [self.max_mel_length, self.n_mels]
            ds = ds.padded_batch(
                batch_size, padded_shapes=padded_shapes, padding_values=padding_values
            )
        elif training:
            # batch items in buckets of similar mel spectrogram length when training
            bucket_boundaries = list(range(0, 10000, 50))
            ds = ds.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    lambda x: x["mel_lengths"],
                    bucket_boundaries=bucket_boundaries,
                    bucket_batch_sizes=[batch_size] * (len(bucket_boundaries) + 1),
                    padded_shapes=padded_shapes,
                    padding_values=padding_values,
                    pad_to_bucket_boundary=False,
                    no_padding=False,
                    drop_remainder=False,
                )
            )
        else:
            ds = ds.padded_batch(
                batch_size, padded_shapes=padded_shapes, padding_values=padding_values
            )
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def __name__(self):
        return "CharactorMelDataset"
