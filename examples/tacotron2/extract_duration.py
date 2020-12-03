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
"""Extract durations based-on tacotron-2 alignments for FastSpeech."""

import argparse
import logging
import os
from numba import jit
import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from examples.tacotron2.tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2


@jit(nopython=True)
def get_duration_from_alignment(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[0])])

    for i in range(np.shape(alignment)[1]):
        max_index = list(alignment[:, i]).index(alignment[:, i].max())
        D[max_index] = D[max_index] + 1

    return D


def main():
    """Running extract tacotron-2 durations."""
    parser = argparse.ArgumentParser(
        description="Extract durations from charactor with trained Tacotron-2 "
        "(See detail in tensorflow_tts/example/tacotron-2/extract_duration.py)."
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
    parser.add_argument("--win-front", default=2, type=int, help="win-front.")
    parser.add_argument("--win-back", default=2, type=int, help="win-front.")
    parser.add_argument(
        "--use-window-mask", default=1, type=int, help="toggle window masking."
    )
    parser.add_argument("--save-alignment", default=0, type=int, help="save-alignment.")
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
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=True,
    )
    dataset = dataset.create(allow_cache=True, batch_size=args.batch_size, drop_remainder=False)

    # define model and load checkpoint
    tacotron2 = TFTacotron2(
        config=Tacotron2Config(**config["tacotron2_params"]),
        name="tacotron2",
    )
    tacotron2._build()  # build model to be able load_weights.
    tacotron2.load_weights(args.checkpoint)

    # apply tf.function for tacotron2.
    tacotron2 = tf.function(tacotron2, experimental_relax_shapes=True)

    for data in tqdm(dataset, desc="[Extract Duration]"):
        utt_ids = data["utt_ids"]
        input_lengths = data["input_lengths"]
        mel_lengths = data["mel_lengths"]
        utt_ids = utt_ids.numpy()
        real_mel_lengths = data["real_mel_lengths"]
        del data["real_mel_lengths"]

        # tacotron2 inference.
        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = tacotron2(
            **data,
            use_window_mask=args.use_window_mask,
            win_front=args.win_front,
            win_back=args.win_back,
            training=True,
        )

        # convert to numpy
        alignment_historys = alignment_historys.numpy()

        for i, alignment in enumerate(alignment_historys):
            real_char_length = input_lengths[i].numpy()
            real_mel_length = real_mel_lengths[i].numpy()
            alignment_mel_length = int(
                np.ceil(
                    real_mel_length / config["tacotron2_params"]["reduction_factor"]
                )
            )
            alignment = alignment[:real_char_length, :alignment_mel_length]
            d = get_duration_from_alignment(alignment)  # [max_char_len]

            d = d * config["tacotron2_params"]["reduction_factor"]
            assert (
                np.sum(d) >= real_mel_length
            ), f"{d}, {np.sum(d)}, {alignment_mel_length}, {real_mel_length}"
            if np.sum(d) > real_mel_length:
                rest = np.sum(d) - real_mel_length
                # print(d, np.sum(d), real_mel_length)
                if d[-1] > rest:
                    d[-1] -= rest
                elif d[0] > rest:
                    d[0] -= rest
                else:
                    d[-1] -= rest // 2
                    d[0] -= rest - rest // 2

                assert d[-1] >= 0 and d[0] >= 0, f"{d}, {np.sum(d)}, {real_mel_length}"

            saved_name = utt_ids[i].decode("utf-8")

            # check a length compatible
            assert (
                len(d) == real_char_length
            ), f"different between len_char and len_durations, {len(d)} and {real_char_length}"

            assert (
                np.sum(d) == real_mel_length
            ), f"different between sum_durations and len_mel, {np.sum(d)} and {real_mel_length}"

            # save D to folder.
            np.save(
                os.path.join(args.outdir, f"{saved_name}-durations.npy"),
                d.astype(np.int32),
                allow_pickle=False,
            )

            # save alignment to debug.
            if args.save_alignment == 1:
                figname = os.path.join(args.outdir, f"{saved_name}_alignment.png")
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.set_title(f"Alignment of {saved_name}")
                im = ax.imshow(
                    alignment, aspect="auto", origin="lower", interpolation="none"
                )
                fig.colorbar(im, ax=ax)
                xlabel = "Decoder timestep"
                plt.xlabel(xlabel)
                plt.ylabel("Encoder timestep")
                plt.tight_layout()
                plt.savefig(figname)
                plt.close()


if __name__ == "__main__":
    main()
