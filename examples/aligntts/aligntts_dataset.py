# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai), Modified by TrinhLQ (@l4zyf9x)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os
import numpy as np
import glob

import tensorflow as tf

from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.processor.vietnamese import text_to_sequence_arpabet, _symbol_to_id


from tqdm import tqdm


class LJSpeechDataset(AbstractDataset):
    def __init__(self,
                 root_dir="/data/trinhlq/dataset/LJSpeech-070720/train",
                 max_mel_length=800
                 ):
        ids_dir = os.path.join(root_dir, "ids")
        mel_dir = os.path.join(root_dir, "raw-feats")
        filenames = [os.path.basename(fn).replace("-ids.npy", "") for fn in glob.glob(f"{ids_dir}/*.npy")]

        self.ids_files = [os.path.join(ids_dir, f"{f}-ids.npy") for f in filenames]
        self.mel_files = [os.path.join(mel_dir, f"{f}-raw-feats.npy") for f in filenames]

        difference = []
        for i in tqdm(range(len(self.ids_files))):
            mel = np.load(self.mel_files[i])
            len_mel = mel.shape[0]
            if len_mel > max_mel_length:
                difference.append(i)
                logging.warning(f"Filter {i}: len_mel({len_mel}) > max_mel_length({max_mel_length})")

        logging.warning(f"After filtering: "
                        f"({len(self.mel_files)} -> {len(self.mel_files)-len(difference)}).")
        self.mel_files = np.delete(self.mel_files, difference)
        self.characters = np.delete(self.ids_files, difference)

    def get_args(self):
        return [self.mel_files]

    def generator(self, mel_files):
        for i, _ in enumerate(mel_files):
            mel = np.load(self.mel_files[i])
            ids = np.load(self.ids_files[i])
            mel_length = mel.shape[0]
            character_length = ids.shape[0]
            speaker_id = 0  # Just hold variable for multispeaker

            items = ids, mel, speaker_id, character_length, mel_length
            yield items

    def create(self,
               allow_cache=False,
               batch_size=1,
               is_shuffle=False,
               map_fn=None,
               reshuffle_each_iteration=True
               ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator,
            output_types=output_types,
            args=(self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(), reshuffle_each_iteration=reshuffle_each_iteration)

        datasets = datasets.padded_batch(
            batch_size,
            padded_shapes=([None], [None, 80],
                           [], [], []),
            padding_values=(0, 0.0,
                            0, 0, 0))
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = (tf.int32, tf.float32,
                        tf.int32, tf.int32, tf.int32)
        return output_types

    def get_len_dataset(self):
        return len(self.mel_files)

    def __name__(self):
        return "LJSpeechDataset"
