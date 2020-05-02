# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode trained FastSpeech from folders."""

import argparse
import logging
import os

import numpy as np
import yaml
import tensorflow as tf

from tqdm import tqdm

from tensorflow_tts.configs import FastSpeechConfig
from tensorflow_tts.datasets import CharactorDurationDataset
from tensorflow_tts.models import TFFastSpeech


def main():
    """Run fastspeech decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Decode soft-mel features from charactor with trained FastSpeech "
                    "(See detail in tensorflow_tts/bin/decode_fastspeech.py).")
    parser.add_argument("--rootdir", default=None, type=str, required=True,
                        help="directory including ids/durations files.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
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
        duration_query = "*-durations.npy"
        char_load_fn = np.load
        duration_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorDurationDataset(
        root_dir=args.rootdir,
        charactor_query=char_query,
        duration_query=duration_query,
        charactor_load_fn=char_load_fn,
        duration_load_fn=duration_load_fn,
        return_utt_id=True
    )
    len_dataset = len(dataset.utt_ids)
    dataset = dataset.create(batch_size=1)

    # define model and load checkpoint
    fastspeech = TFFastSpeech(config=FastSpeechConfig(**config["fastspeech_params"]), name='fastspeech')
    fastspeech._build()
    fastspeech.load_weights(args.checkpoint)

    fastspeech = tf.function(fastspeech, experimental_relax_shapes=True)

    pbar = tqdm(initial=0,
                total=len_dataset,
                desc="[Decoding]]")

    for data in dataset:
        utt_id = data[0].numpy().decode("utf-8")
        char_id = data[1]
        duration = data[2]

        # expand input
        ids = tf.expand_dims(char_id, 0)
        durations = tf.expand_dims(duration, 0)

        # fastspeech inference.
        masked_mel_before, masked_mel_after, _ = fastspeech(
            ids,
            attention_mask=tf.math.not_equal(ids, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(ids)[0]]),
            duration_gts=durations,
            training=False
        )

        # convert to numpy
        masked_mel_before = masked_mel_before.numpy()
        masked_mel_after = masked_mel_after.numpy()

        # save to folder.
        np.save(os.path.join(args.outdir, f"{utt_id}-fs-before-feats.npy"),
                masked_mel_before.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(args.outdir, f"{utt_id}-fs-after-feats.npy"),
                masked_mel_after.astype(np.float32), allow_pickle=False)
        # update progress bar.
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
