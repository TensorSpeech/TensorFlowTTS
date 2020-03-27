# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import tensorflow as tf
import numpy as np
import math

from tensorflow_tts.utils import find_files
from tensorflow_tts.utils import read_hdf5


class AudioMelDataset(tf.keras.utils.Sequence):
    """Tensorflow compatible audio and mel dataset."""

    def __init__(self,
                 root_dir,
                 batch_size=1,
                 audio_query="*.h5",
                 mel_query="*.h5",
                 audio_load_fn=lambda x: read_hdf5(x, "wave"),
                 mel_load_fn=lambda x: read_hdf5(x, "feats"),
                 audio_length_threshold=None,
                 mel_length_threshold=None,
                 return_utt_id=False,
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

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [idx for idx in range(len(mel_files)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(mel_files), \
            f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        self.batch_size = batch_size

        if ".npy" in audio_query:
            self.utt_ids = [os.path.basename(f).replace("-wave.npy", "") for f in audio_files]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        self.return_utt_id = return_utt_id

    def __len__(self):
        return math.ceil(len(self.audio_files) / self.batch_size)

    def __getitem__(self, idx):
        utt_ids = self.utt_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        audio_files = self.audio_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        mel_files = self.mel_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        # map function
        utt_ids = list(map(lambda x: x, utt_ids))  # [B]
        audios = list(map(self.audio_load_fn, audio_files))  # [B, T(dynamic)]
        mels = list(map(self.mel_load_fn, mel_files))  # [B, T'(dynamic), C]

        if self.return_utt_id:
            items = utt_ids, audios, mels
        else:
            items = audios, mels

        return items

    def __name__(self):
        return "Sequence"


class AudioDataset(tf.keras.utils.Sequence):
    """Tensorflow compatible audio dataset"""

    def __init__(self,
                 root_dir,
                 batch_size=1,
                 audio_query="*.h5",
                 audio_load_fn=lambda x: read_hdf5(x, "wave"),
                 audio_length_threshold=None,
                 return_utt_id=False,
                 ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of audio and mel files.
        audio_files = sorted(find_files(root_dir, audio_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.batch_size = batch_size
        self.return_utt_id = return_utt_id

        if ".npy" in audio_query:
            self.utt_ids = [os.path.basename(f).replace("-wave.npy", "") for f in audio_files]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]

    def __len__(self):
        return math.ceil(len(self.audio_files) / self.batch_size)

    def __getitem__(self, idx):
        utt_ids = self.utt_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        audio_files = self.audio_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        # map function
        utt_ids = list(map(lambda x: x, utt_ids))  # [B]
        audios = list(map(self.audio_load_fn, audio_files))  # [B, T(dynamic)]

        if self.return_utt_id:
            items = utt_ids, audios
        else:
            items = audios

        return items

    def __name__(self):
        return "Sequence"


class MelDataset(tf.keras.utils.Sequence):
    """Tensorflow compatible mel dataset"""

    def __init__(self,
                 root_dir,
                 batch_size=1,
                 mel_query="*.h5",
                 mel_load_fn=lambda x: read_hdf5(x, "feats"),
                 mel_length_threshold=None,
                 return_utt_id=False,
                 ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            
        """
        # find all of audio and mel files.
        mel_files = sorted(find_files(root_dir, mel_query))
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(mel_files[0:2])

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [idx for idx in range(len(mel_files)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_files) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_files)} -> {len(idxs)}).")
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.batch_size = batch_size

        if ".npy" in mel_query:
            self.utt_ids = [os.path.basename(f).replace("-feats.npy", "") for f in mel_files]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]

    def __len__(self):
        return math.ceil(len(self.mel_files) / self.batch_size)

    def __getitem__(self, idx):
        utt_ids = self.utt_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        mel_files = self.mel_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        # map function
        utt_ids = list(map(lambda x: x, utt_ids))  # [B]
        mels = list(map(self.mel_load_fn, mel_files))  # [B, T'(dynamic), C]

        if self.return_utt_id:
            items = utt_ids, mels
        else:
            items = mels

        return items

    def __name__(self):
        return "Sequence"
