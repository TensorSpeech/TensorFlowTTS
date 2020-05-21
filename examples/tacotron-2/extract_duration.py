# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

"""Extract durations based-on tacotron-2 alignments for FastSpeech."""

import argparse
import logging
import os

import numpy as np
import yaml
import tensorflow as tf

from tqdm import tqdm

from tensorflow_tts.configs import Tacotron2Config
from tacotron_dataset import CharactorMelDataset
from tensorflow_tts.models import TFTacotron2

import matplotlib.pyplot as plt


def get_duration_from_alignment(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[0])])

    for i in range(np.shape(alignment)[1]):
        max_index = alignment[:, i].tolist().index(alignment[:, i].max())
        D[max_index] = D[max_index] + 1

    return D


def main():
    """Run extract tacotron-2 durations."""
    parser = argparse.ArgumentParser(
        description="Extract durations from charactor with trained Tacotron-2 "
                    "(See detail in tensorflow_tts/example/tacotron-2/extract_duration.py).")
    parser.add_argument("--rootdir", default=None, type=str, required=True,
                        help="directory including ids/durations files.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--use-norm", default=1, type=int,
                        help="usr norm-mels for train or raw.")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="batch size.")
    parser.add_argument("--win-front", default=2, type=int,
                        help="win-front.")
    parser.add_argument("--win-back", default=2, type=int,
                        help="win-front.")
    parser.add_argument("--save-alignment", default=0, type=int,
                        help="save-alignment.")
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
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        char_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorMelDataset(
        root_dir=args.rootdir,
        charactor_query=char_query,
        mel_query=mel_query,
        charactor_load_fn=char_load_fn,
        mel_load_fn=mel_load_fn,
        return_utt_id=True
    )
    dataset = dataset.create(batch_size=args.batch_size)

    # define model and load checkpoint
    tacotron2 = TFTacotron2(config=Tacotron2Config(**config["tacotron2_params"]),
                            training=True,  # enable teacher forcing mode.
                            name='tacotron2')
    tacotron2._build()  # build model to be able load_weights.
    tacotron2.load_weights(args.checkpoint)

    tacotron2 = tf.function(tacotron2, experimental_relax_shapes=True)

    for data in tqdm(dataset, desc="[Extraction]"):
        utt_id, charactor, char_length, mel, mel_length, g_attention = data
        utt_id = utt_id.numpy()

        # tacotron2 inference.
        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = tacotron2(
            charactor,
            char_length,
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            mel_outputs=mel,
            mel_lengths=mel_length,
            use_window_mask=True,
            win_front=args.win_front,
            win_back=args.win_back,
            training=True
        )

        # convert to numpy
        alignment_historys = alignment_historys.numpy()

        for i, alignment in enumerate(alignment_historys):
            d = get_duration_from_alignment(alignment)  # [max_char_len]
            real_length = char_length[i].numpy() - 1  # minus 1 because char have eos tokens.
            d_real = d[:real_length]

            saved_name = utt_id[i].decode("utf-8")

            # save D to folder.
            np.save(os.path.join(args.outdir, f"{saved_name}-durations.npy"),
                    d_real.astype(np.int32), allow_pickle=False)

            # save alignment to debug.
            if args.save_alignment is True:
                figname = os.path.join(args.outdir, f"{saved_name}_alignment.png")
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.set_title(f'Alignment of {saved_name}')
                im = ax.imshow(
                    alignment[:real_length, :mel_length[i].numpy()],
                    aspect='auto',
                    origin='lower',
                    interpolation='none')
                fig.colorbar(im, ax=ax)
                xlabel = 'Decoder timestep'
                plt.xlabel(xlabel)
                plt.ylabel('Encoder timestep')
                plt.tight_layout()
                plt.savefig(figname)
                plt.close()


if __name__ == "__main__":
    main()
