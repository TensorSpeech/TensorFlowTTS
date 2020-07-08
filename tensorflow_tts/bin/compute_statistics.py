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
"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tensorflow_tts.datasets import MelDataset
from tensorflow_tts.datasets import AudioDataset

from tensorflow_tts.utils import remove_outlier

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of dumped raw features "
        "(See detail in tensorflow_tts/bin/compute_statistics.py)."
    )
    parser.add_argument(
        "--rootdir", type=str, required=True, help="directory including feature files. "
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
        help="directory to save statistics.",
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
    if args.outdir is None:
        args.outdir = os.path.dirname(args.rootdir)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # get dataset
    if config["format"] == "npy":
        mel_query = "*-raw-feats.npy"
        f0_query = "*-raw-f0.npy"
        energy_query = "*-raw-energy.npy"
        mel_load_fn = np.load
    else:
        raise ValueError("Support only npy format.")

    dataset = MelDataset(
        args.rootdir, mel_query=mel_query, mel_load_fn=mel_load_fn
    ).create(batch_size=1)

    # calculate statistics
    scaler = StandardScaler()
    for mel, mel_length in tqdm(dataset):
        mel = mel[0].numpy()
        scaler.partial_fit(mel)

    # save to file
    stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
    np.save(
        os.path.join(args.outdir, "stats.npy"),
        stats.astype(np.float32),
        allow_pickle=False,
    )

    # calculate statistic of f0
    f0_dataset = AudioDataset(
        args.rootdir, audio_query=f0_query, audio_load_fn=np.load,
    ).create(batch_size=1)

    pitch_vecs = []
    for f0, f0_length in tqdm(f0_dataset):
        f0 = f0[0].numpy()  # [T]
        f0 = remove_outlier(f0)
        pitch_vecs.append(f0)
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in pitch_vecs])
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    # save to file
    stats = np.stack([mean, std], axis=0)
    np.save(
        os.path.join(args.outdir, "stats_f0.npy"),
        stats.astype(np.float32),
        allow_pickle=False,
    )

    # calculate statistic of energy
    energy_dataset = AudioDataset(
        args.rootdir, audio_query=energy_query, audio_load_fn=np.load,
    ).create(batch_size=1)

    energy_vecs = []
    for e, e_length in tqdm(energy_dataset):
        e = e[0].numpy()
        e = remove_outlier(e)
        energy_vecs.append(e)
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in energy_vecs])
    mean, std = np.mean(nonzeros), np.std(nonzeros)

    # save to file
    stats = np.stack([mean, std], axis=0)
    np.save(
        os.path.join(args.outdir, "stats_energy.npy"),
        stats.astype(np.float32),
        allow_pickle=False,
    )


if __name__ == "__main__":
    main()
