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
"""Decode trained LightSpeech from folders."""

import argparse
import logging
import os
import sys

sys.path.append(".")

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from TensorFlowTTS.examples.fastspeech.fastspeech_dataset import CharactorDataset
from tensorflow_tts.configs import LightSpeechConfig
from tensorflow_tts.models import TFLightSpeech


def main():
    """Run lightspeech decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Decode soft-mel features from charactor with trained LightSpeech "
        "(See detail in examples/lightspeech/decode_lightspeech.py)."
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
        "--config",
        default=None,
        type=str,
        required=True,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        required=False,
        help="Batch size for inference.",
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
        char_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorDataset(
        root_dir=args.rootdir,
        charactor_query=char_query,
        charactor_load_fn=char_load_fn,
    )
    dataset = dataset.create(batch_size=args.batch_size)

    # define model and load checkpoint
    lightspeech = TFLightSpeech(
        config=LightSpeechConfig(**config["lightspeech_params"]), name="lightspeech"
    )
    lightspeech._build()
    lightspeech.load_weights(args.checkpoint)

    for data in tqdm(dataset, desc="Decoding"):
        utt_ids = data["utt_ids"]
        char_ids = data["input_ids"]

        # lightspeech inference.
        (masked_mel, duration_outputs, _) = lightspeech.inference(
            char_ids,
            speaker_ids=tf.zeros(shape=[tf.shape(char_ids)[0]], dtype=tf.int32),
            speed_ratios=tf.ones(shape=[tf.shape(char_ids)[0]], dtype=tf.float32),
            f0_ratios=tf.ones(shape=[tf.shape(char_ids)[0]], dtype=tf.float32),
        )

        # convert to numpy
        masked_mels = masked_mel.numpy()

        for (utt_id, mel, durations) in zip(utt_ids, masked_mels, duration_outputs):
            # real len of mel predicted
            real_length = durations.numpy().sum()
            utt_id = utt_id.numpy().decode("utf-8")
            # save to folder.
            np.save(
                os.path.join(args.outdir, f"{utt_id}-ls-feats.npy"),
                mel[:real_length, :].astype(np.float32),
                allow_pickle=False,
            )


if __name__ == "__main__":
    main()
