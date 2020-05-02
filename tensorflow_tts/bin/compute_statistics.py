# -*- coding: utf-8 -*-

# This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""Calculate statistics of feature files."""

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
        description="Compute mean and variance of dumped raw features "
                    "(See detail in tensorflow_tts/bin/compute_statistics.py).")
    parser.add_argument("--rootdir", type=str, required=True,
                        help="directory including feature files. ")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--outdir", default=None, type=str, required=True,
                        help="directory to save statistics.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

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
        mel_load_fn = np.load
    else:
        raise ValueError("Support only npy format.")

    dataset = MelDataset(
        args.rootdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn
    ).create(batch_size=1)

    # calculate statistics
    scaler = StandardScaler()
    for mel in tqdm(dataset):
        mel = mel[0].numpy()
        scaler.partial_fit(mel)

    # save to file
    stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
    np.save(os.path.join(args.outdir, "stats.npy"), stats.astype(np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
