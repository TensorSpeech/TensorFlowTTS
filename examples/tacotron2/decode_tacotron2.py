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
"""Decode Tacotron-2."""

import argparse
import logging
import os
import sys

sys.path.append(".")

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from examples.tacotron2.tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2


def main():
    """Running decode tacotron-2 mel-spectrogram."""
    parser = argparse.ArgumentParser(
        description="Decode mel-spectrogram from folder ids with trained Tacotron-2 "
        "(See detail in tensorflow_tts/example/tacotron2/decode_tacotron2.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="directory including ids/durations files.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save generated speech."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size.")
    parser.add_argument("--win-front", default=3, type=int, help="win-front.")
    parser.add_argument("--win-back", default=3, type=int, help="win-front.")
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        required=True,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    if config["format"] == "npy":
        char_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        char_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=args.rootdir,
        charactor_query=char_query,
        mel_query=mel_query,
        charactor_load_fn=char_load_fn,
        mel_load_fn=mel_load_fn,
        reduction_factor=config["tacotron2_params"]["reduction_factor"]
    )
    dataset = dataset.create(allow_cache=True, batch_size=args.batch_size)

    # define model and load checkpoint
    tacotron2 = TFTacotron2(
        config=Tacotron2Config(**config["tacotron2_params"]),
        name="tacotron2",
    )
    tacotron2._build()  # build model to be able load_weights.
    tacotron2.load_weights(args.checkpoint)

    # setup window
    tacotron2.setup_window(win_front=args.win_front, win_back=args.win_back)

    for data in tqdm(dataset, desc="[Decoding]"):
        utt_ids = data["utt_ids"]
        utt_ids = utt_ids.numpy()

        # tacotron2 inference.
        (
            mel_outputs,
            post_mel_outputs,
            stop_outputs,
            alignment_historys,
        ) = tacotron2.inference(
            input_ids=data["input_ids"], 
            input_lengths=data["input_lengths"], 
            speaker_ids=data["speaker_ids"],
        )

        # convert to numpy
        post_mel_outputs = post_mel_outputs.numpy()

        for i, post_mel_output in enumerate(post_mel_outputs):
            stop_token = tf.math.round(tf.nn.sigmoid(stop_outputs[i]))  # [T]
            real_length = tf.math.reduce_sum(
                tf.cast(tf.math.equal(stop_token, 0.0), tf.int32), -1
            )
            post_mel_output = post_mel_output[:real_length, :]

            saved_name = utt_ids[i].decode("utf-8")

            # save D to folder.
            np.save(
                os.path.join(args.outdir, f"{saved_name}-norm-feats.npy"),
                post_mel_output.astype(np.float32),
                allow_pickle=False,
            )


if __name__ == "__main__":
    main()
