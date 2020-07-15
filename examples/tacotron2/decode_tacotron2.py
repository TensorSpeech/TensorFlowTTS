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
"""Decode Tacotron 2."""

import argparse
import glob
import logging
import os

import yaml

import numpy as np
import tensorflow as tf

from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.processor.ljspeech import LJSpeechProcessor, symbols
from tqdm import tqdm


def build_ds(generator, features, config):
    """Build tf.data.Dataset for decoding.
    Args:
        generator: builds output to be fed to model for decoding.
        features (list): arguments for the generator function containing data items.
        config (dict): Config dict loaded from YAML format configuration file.
    Return:
        ds (tf.data.Dataset): dataset with padded elements and transformations.
    """
    ds = tf.data.Dataset.from_generator(
        generator,
        output_types={
            "input_ids": tf.int32,
            "input_lengths": tf.int32,
            "speaker_ids": tf.int32,
            "utt_ids": tf.string,
        },
        args=[features],
    )
    padding_values = {
        "input_ids": 0,
        "input_lengths": 0,
        "speaker_ids": 0,
        "utt_ids": "",
    }
    padded_shapes = {
        "input_ids": [None],
        "input_lengths": [],
        "speaker_ids": [],
        "utt_ids": [],
    }
    ds = ds.map(
        lambda item: {
            "input_ids": tf.concat([item["input_ids"], [len(symbols) - 1]], -1),
            "input_lengths": item["input_lengths"] + 1,
            "speaker_ids": item["speaker_ids"],
            "utt_ids": item["utt_ids"],
        },
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.padded_batch(
        config["batch_size"], padded_shapes=padded_shapes, padding_values=padding_values
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def main():
    """Running decode Tacotron 2 mel spectrogram."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode mel spectrogram from folder IDs with trained Tacotron 2 "
            "(See details in tensorflow_tts/example/tacotron2/decode_tacotron2.py)."
        )
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        required=True,
        help=(
            "Directory including IDs/durations files, or path to text file containing"
            " sentences."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ljspeech",
        choices=["ljspeech"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to save generated speech."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint file to be loaded."
    )
    parser.add_argument(
        "--inverse_norm",
        action="store_true",
        help="Whether or not to denormalize features.",
    )
    parser.add_argument(
        "--stats_path",
        default=argparse.SUPPRESS,
        type=str,
        help=(
            "Path to the statistics file with mean and std values for standardization."
        ),
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument(
        "--win_front", default=3, type=int, help="Window to apply in the front."
    )
    parser.add_argument(
        "--win_back", default=3, type=int, help="Window to apply in the back."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "YAML format configuration file. If not explicitly provided, it will be"
            " searched in the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO, 2: WARN.",
    )
    args = parser.parse_args()

    # load config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(vars(args))

    if config["inverse_norm"] and "stats_path" not in config:
        raise ValueError(
            "'--stats_path' should be provided when using '--inverse_norm'."
        )
    if config["inverse_norm"]:
        mel_mean, mel_scale = np.load(config["stats_path"])
        mel_save_name = "raw-feats"
    else:
        config["stats_path"] = None
        mel_save_name = "norm-feats"

    # set logger
    fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN}
    logging.basicConfig(level=log_level[config["verbose"]], format=fmt)

    # check directory existence
    os.makedirs(config["outdir"], exist_ok=True)

    if os.path.isdir(config["rootdir"]):
        char_files_path = os.path.join(config["rootdir"], "ids", "*-ids.npy")
        char_files = sorted(glob.glob(char_files_path))
        assert len(char_files) > 0, f"No files found in '{char_files_path}'."

        utt_ids = [os.path.basename(x)[:10] for x in char_files]
        features = list(zip(utt_ids, char_files))

        def generator(features):
            for utt_id, char_file, in features:
                char = np.load(char_file)
                yield {
                    "input_ids": char,
                    "input_lengths": char.shape[0],
                    "speaker_ids": 0,
                    "utt_ids": utt_id,
                }

    elif os.path.splitext(config["rootdir"])[1] == ".txt":
        with open(config["rootdir"], encoding="utf8") as f:
            sentences = f.read().splitlines()
        assert len(sentences) > 0, f"No text found in '{config['rootdir']}'."
        dataset_processor = {
            "ljspeech": LJSpeechProcessor,
        }
        processor = dataset_processor[config["dataset"]](
            None, cleaner_names="english_cleaners"
        )
        features = [processor.text_to_sequence(line) for line in sentences]
        # TODO: find a better workaround for "Can't convert non-rectangular Python sequence to Tensor"
        features = [",".join(map(str, line)) for line in features]

        def generator(features):
            for utt_id, char in enumerate(features, 1):
                char = np.asarray(char.split(",".encode("utf8")), np.int32)
                yield {
                    "input_ids": char,
                    "input_lengths": char.shape[0],
                    "speaker_ids": 0,
                    "utt_ids": str(utt_id).zfill(4),
                }

    else:
        raise ValueError("Only 'npy' directories and 'txt' files are supported.")

    ds = build_ds(generator, features, config)

    # define model and load checkpoint
    tacotron2 = TFTacotron2(
        config=Tacotron2Config(**config["tacotron2_params"]),
        training=False,  # disable teacher forcing mode
        name="tacotron2",
    )
    tacotron2._build()  # build model to be able load weights
    tacotron2.load_weights(config["checkpoint"])

    # setup window
    tacotron2.setup_window(win_front=config["win_front"], win_back=config["win_back"])

    for batch in tqdm(ds, desc="[Decoding]"):
        bs, _ = tf.shape(batch["input_ids"])
        # tacotron2 inference
        _, mel_preds, stop_preds, _ = tacotron2.inference(
            batch["input_ids"],
            batch["input_lengths"],
            speaker_ids=batch["speaker_ids"],
        )
        stop_token = ~tf.cast(tf.math.round(tf.nn.sigmoid(stop_preds)), tf.bool)
        mel_preds = tf.ragged.boolean_mask(mel_preds, stop_token)

        for idx, mel_pred in enumerate(mel_preds):
            utt_id_str = batch["utt_ids"][idx].numpy().decode("utf8")
            if config["inverse_norm"]:
                mel_pred = mel_pred * mel_scale + mel_mean
            np.save(
                os.path.join(config["outdir"], f"{utt_id_str}-{mel_save_name}.npy"),
                mel_pred.numpy().astype(np.float32),
                allow_pickle=False,
            )


if __name__ == "__main__":
    main()
