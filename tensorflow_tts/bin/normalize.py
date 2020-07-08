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

"""Normalize feature files and dump them."""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tensorflow_tts.datasets import MelDataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Normalize dumped raw features (See detail in tensorflow_tts/bin/normalize.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="directory including feature files to be normalized. ",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to dump normalized feature files.",
    )
    parser.add_argument("--stats", type=str, required=True, help="statistics file.")
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
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

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check directory existence
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "train", "norm-feats"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "valid", "norm-feats"), exist_ok=True)

    # get dataset
    if args.rootdir is not None:
        if config["format"] == "npy":
            mel_query = "*-raw-feats.npy"

            def mel_load_fn(x):
                return np.load(x, allow_pickle=True)

        else:
            raise ValueError("support only npy format.")

        dataset = MelDataset(
            args.rootdir,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            return_utt_id=True,
        ).create(batch_size=1)

    # restore scaler
    scaler = StandardScaler()
    if config["format"] == "npy":
        scaler.mean_ = np.load(args.stats)[0]
        scaler.scale_ = np.load(args.stats)[1]
        scaler.n_features_in_ = config["num_mels"]
    else:
        raise ValueError("Support only npy format")

    # load train/valid utt_ids
    train_utt_ids = np.load(os.path.join(args.rootdir, "train_utt_ids.npy"))
    valid_utt_ids = np.load(os.path.join(args.rootdir, "valid_utt_ids.npy"))

    # process each file
    for items in tqdm(dataset):
        utt_id, mel, _ = items

        # convert to numpy
        utt_id = utt_id[0].numpy().decode("utf-8")
        mel = mel[0].numpy()

        # normalize
        mel = scaler.transform(mel)

        # save
        if config["format"] == "npy":
            if utt_id in train_utt_ids:
                subdir = "train"
            elif utt_id in valid_utt_ids:
                subdir = "valid"
            np.save(
                os.path.join(
                    args.outdir, subdir, "norm-feats", f"{utt_id}-norm-feats.npy"
                ),
                mel.astype(np.float32),
                allow_pickle=False,
            )
        else:
            raise ValueError("support only npy format.")


if __name__ == "__main__":
    main()
