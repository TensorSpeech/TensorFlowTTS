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
"""Decode trained FastSpeech from folders."""

import argparse
import logging
import os
import sys
import json

sys.path.append(".")

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from TensorFlowTTS.examples.fastspeech2_libritts.fastspeech2_dataset import (
    CharactorDurationF0EnergyMelDataset,
)
from tensorflow_tts.configs import LightSpeechConfig
from tensorflow_tts.models import TFLightSpeech


def main():
    """Run lightspeech decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Decode soft-mel features from charactor with trained FastSpeech "
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
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--dataset_mapping",
        default="dump/libritts_mapper.npy",
        type=str,
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

    outdpost = os.path.join(args.outdir, "postnets")

    if not os.path.exists(outdpost):
        os.makedirs(outdpost)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    if config["format"] == "npy":
        char_query = "*-ids.npy"
        char_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    with open(args.dataset_mapping) as f:
        dataset_mapping = json.load(f)
        speakers_map = dataset_mapping["speakers_map"]

    # Check n_speakers matches number of speakers in speakers_map
    n_speakers = config["lightspeech_params"]["n_speakers"]
    assert n_speakers == len(
        speakers_map
    ), f"Number of speakers in dataset does not match n_speakers in config"

    # define data-loader
    dataset = CharactorDurationF0EnergyMelDataset(
        root_dir=args.rootdir,
        charactor_query=char_query,
        charactor_load_fn=char_load_fn,
        f0_stat=f"./{args.rootdir.split('/')[-2]}/stats_f0.npy",
        energy_stat=f"./{args.rootdir.split('/')[-2]}/stats_energy.npy",
        speakers_map=speakers_map,
    )
    dataset = dataset.create(
        batch_size=1
    )  # force batch size to 1 otherwise it may miss certain files

    # define model and load checkpoint
    lightspeech = TFLightSpeech(
        config=LightSpeechConfig(**config["lightspeech_params"]), name="lightspeech"
    )
    lightspeech._build()
    lightspeech.load_weights(args.checkpoint)
    lightspeech = tf.function(lightspeech, experimental_relax_shapes=True)

    for data in tqdm(dataset, desc="Decoding"):
        utt_ids = data["utt_ids"]
        mel_lens = data["mel_lengths"]

        # lightspeech inference.
        masked_mel_before, masked_mel_after, duration_outputs, _ = lightspeech(
            **data, training=True
        )

        # convert to numpy
        masked_mel_befores = masked_mel_before.numpy()
        masked_mel_afters = masked_mel_after.numpy()

        for (utt_id, _, mel_after, _, mel_len) in zip(
            utt_ids, masked_mel_befores, masked_mel_afters, duration_outputs, mel_lens
        ):
            utt_id = utt_id.numpy().decode("utf-8")

            np.save(
                os.path.join(outdpost, f"{utt_id}-postnet.npy"),
                mel_after[:mel_len, :].astype(np.float32),
                allow_pickle=False,
            )


if __name__ == "__main__":
    main()
