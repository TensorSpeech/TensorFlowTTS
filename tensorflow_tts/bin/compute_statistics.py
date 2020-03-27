# -*- coding: utf-8 -*-

# This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import tensorflow as tf
import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tensorflow_tts.dataset import MelSCPDataset
from tensorflow_tts.dataset import MelDataset
from tensorflow_tts.utils import read_hdf5
from tensorflow_tts.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of dumped raw features "
                    "(See detail in parallel_wavegan/bin/compute_statistics.py).")
    parser.add_argument("--feats-scp", "--scp", default=None, type=str,
                        help="kaldi-style feats.scp file. "
                             "you need to specify either feats-scp or rootdir.")
    parser.add_argument("--rootdir", type=str, required=True,
                        help="directory including feature files. "
                             "you need to specify either feats-scp or rootdir.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory to save statistics. if not provided, "
                             "stats will be saved in the above root directory. (default=None)")
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

    # check arguments
    if (args.feats_scp is not None and args.rootdir is not None) or \
            (args.feats_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if args.dumpdir is None:
        args.dumpdir = os.path.dirname(args.rootdir)
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if args.feats_scp is None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, "feats")
        elif config["format"] == "npy":
            mel_query = "*-feats.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("Support only hdf5 or npy format.")

        # TODO(@dathudeptrai), use tf.data rather than tf.keras.utils.Sequence
        # and Support batch_size != 1
        dataset = MelDataset(
            args.rootdir,
            batch_size=1,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn
        )
    else:
        dataset = MelSCPDataset(args.feats_scp)
        dataset = dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    logging.info(f"The number of files = {len(dataset)}.")

    # calculate statistics
    scaler = StandardScaler()
    for mel in tqdm(dataset):
        # convert to numpy if dataset is instance of tf.data
        if "Dataset" in dataset.__name__():
            mel = mel[0].numpy()
        else:
            mel = mel[0]

        scaler.partial_fit(mel)

    if config["format"] == "hdf5":
        write_hdf5(os.path.join(args.dumpdir, "stats.h5"), "mean", scaler.mean_.astype(np.float32))
        write_hdf5(os.path.join(args.dumpdir, "stats.h5"), "scale", scaler.scale_.astype(np.float32))
    else:
        stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
        np.save(os.path.join(args.dumpdir, "stats.npy"), stats.astype(np.float32), allow_pickle=False)


if __name__ == "__main__":
    main()
