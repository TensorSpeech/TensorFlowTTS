# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules based on kaldi-style scp files and tf.data.Dataset."""

import logging

import numpy as np
import kaldiio

import tensorflow as tf


class AudioMelSCPDataset(tf.data.Dataset):
    """Tensorflow compatible audio and mel dataset based on kaldi-stype scp files."""

    def __new__(cls,
                wav_scp,
                feats_scp,
                segments=None,
                audio_length_threshold=None,
                mel_length_threshold=None,
                is_shuffle=False,
                return_utt_id=False,
                return_sampling_rate=False,
                allow_cache=False,
                batch_size=1,
                map_fn=None):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            feats_scp (str): Kaldi-style fests.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wav_scp, segments=segments)
        mel_loader = kaldiio.load_scp(feats_scp)
        audio_keys = list(audio_loader.keys())
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [idx for idx in range(len(audio_keys)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_keys) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]

        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [idx for idx in range(len(mel_keys)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_keys) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]

        # assert the number of files
        warning_len = f"Number of audio and mel files are different ({len(audio_keys)} vs {len(mel_keys)})."
        assert len(audio_keys) == len(mel_keys), warning_len

        def generator(return_sampling_rate,
                      return_utt_id):
            for k in audio_loader:
                if k in audio_keys:
                    fs, audio = audio_loader[k]
                    mel = mel_loader[k]

                    # normalize audio signal to be [-1, 1]
                    audio = audio.astype(np.float32)
                    audio /= (1 << (16 - 1))  # assume that wav is PCM 16 bit

                    if return_sampling_rate:
                        audio = (audio, np.int32(fs))

                    if return_utt_id:
                        items = (k, *audio, mel)
                    else:
                        items = (audio, mel)
                    yield items

        output_types = (tf.float32,)
        if return_sampling_rate:
            output_types = (*output_types, tf.int32)
        if return_utt_id:
            output_types = (tf.dtypes.string, *output_types, tf.float32)

        audio_mel_datasets = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            args=(return_sampling_rate,
                  return_utt_id)
        )

        if allow_cache:
            audio_mel_datasets = audio_mel_datasets.cache()

        if is_shuffle:
            audio_mel_datasets = audio_mel_datasets.shuffle(
                len(audio_keys), reshuffle_each_iteration=True
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            audio_mel_datasets = audio_mel_datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        audio_mel_datasets = audio_mel_datasets.batch(batch_size)
        audio_mel_datasets = audio_mel_datasets.prefetch(tf.data.experimental.AUTOTUNE)

        return audio_mel_datasets

    def __name__(self):
        return "Dataset"


class AudioSCPDataset(tf.data.Dataset):
    """Tensorflow compatible audio dataset based on kaldi-stype scp files."""

    def __new__(cls,
                wav_scp,
                segments=None,
                audio_length_threshold=None,
                return_utt_id=False,
                return_sampling_rate=False,
                is_shuffle=False,
                allow_cache=False,
                batch_size=1,
                map_fn=None):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wav_scp, segments=segments)
        audio_keys = list(audio_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [idx for idx in range(len(audio_keys)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_keys) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_keys)} -> {len(idxs)}).")
            audio_keys = [audio_keys[idx] for idx in idxs]

        def generator(return_sampling_rate,
                      return_utt_id):
            for k in audio_loader:
                if k in audio_keys:
                    (fs, audio) = audio_loader[k]

                    # normalize audio signal to be [-1, 1]
                    audio = audio.astype(np.float32)
                    audio /= (1 << (16 - 1))  # assume that wav is PCM 16 bit

                    if return_sampling_rate:
                        audio = (audio, np.int32(fs))

                    if return_utt_id:
                        items = (k, *audio)
                    else:
                        items = (audio,)
                    yield items

        output_types = (tf.float32,)
        if return_sampling_rate:
            output_types = (*output_types, tf.int32)
        if return_utt_id:
            output_types = (tf.dtypes.string, *output_types)

        audio_datasets = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            args=(return_sampling_rate,
                  return_utt_id,)
        )

        if allow_cache:
            audio_datasets = audio_datasets.cache()

        if is_shuffle:
            audio_datasets = audio_datasets.shuffle(
                len(audio_keys), reshuffle_each_iteration=True
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            audio_datasets = audio_datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        audio_datasets = audio_datasets.batch(batch_size)
        audio_datasets = audio_datasets.prefetch(tf.data.experimental.AUTOTUNE)

        return audio_datasets

    def __name__(self):
        return "Dataset"


class MelSCPDataset(tf.data.Dataset):
    """Tensorflow compatible mel dataset based on kaldi-stype scp files."""

    def __new__(cls,
                feats_scp,
                mel_length_threshold=None,
                return_utt_id=False,
                is_shuffle=False,
                allow_cache=False,
                batch_size=1,
                map_fn=None):
        """Initialize dataset.

        Args:
            feats_scp (str): Kaldi-style fests.scp file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.

        """
        # load scp as lazy dict
        mel_loader = kaldiio.load_scp(feats_scp)
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [idx for idx in range(len(mel_keys)) if mel_lengths[idx] > mel_length_threshold]
            if len(mel_keys) != len(idxs):
                logging.warning(f"Some files are filtered by mel length threshold "
                                f"({len(mel_keys)} -> {len(idxs)}).")
            mel_keys = [mel_keys[idx] for idx in idxs]

        def generator(return_utt_id):
            for k in mel_loader:
                if k in mel_keys:
                    mel = mel_loader[k]

                    if return_utt_id:
                        items = (k, mel)
                    else:
                        items = (mel,)
                    yield items

        output_types = (tf.float32,) if return_utt_id is False else (tf.dtypes.string, tf.float32)
        mel_datasets = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            args=(return_utt_id)
        )

        if allow_cache:
            mel_datasets = mel_datasets.cache()

        if is_shuffle:
            mel_datasets = mel_datasets.shuffle(
                len(mel_keys), reshuffle_each_iteration=True
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            mel_datasets = mel_datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        mel_datasets = mel_datasets.batch(batch_size)
        mel_datasets = mel_datasets.prefetch(tf.data.experimental.AUTOTUNE)

        return mel_datasets

    def __name__(self):
        return "Dataset"
