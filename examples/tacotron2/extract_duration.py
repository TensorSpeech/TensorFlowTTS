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
"""Extract durations based on Tacotron 2 alignments for FastSpeech."""

import argparse
import logging
import os

import yaml

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

from tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.models import TFTacotron2


def main():
    """Running extract Tacotron 2 durations."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract durations from characters with trained Tacotron 2 (See detail in"
            " tensorflow_tts/example/tacotron-2/extract_duration.py)."
        )
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        required=True,
        help="Directory including ids/durations files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where generated speech will be saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint file to be loaded."
    )
    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="Whether or not to use normalized features.",
    )
    parser.add_argument(
        "--stats_path",
        default=argparse.SUPPRESS,
        type=str,
        help=(
            "Path to the statistics file with mean and std values for standardization."
        ),
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument(
        "--win_front", default=3, type=int, help="Window to apply in the front."
    )
    parser.add_argument(
        "--win_back", default=3, type=int, help="Window to apply in the back."
    )
    parser.add_argument(
        "--save_alignment",
        action="store_true",
        help="Whether or not to save alignment plot.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "YAML format configuration file. If not explicitly provided, it will be"
            " searched in the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO, 2: WARN.",
    )
    args = parser.parse_args()

    # load config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(vars(args))

    # set logger
    fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN}
    logging.basicConfig(level=log_level[config["verbose"]], format=fmt)

    # check directory existence
    os.makedirs(config["outdir"], exist_ok=True)

    if config["use_norm"] and "stats_path" not in config:
        raise ValueError("'--stats_path' should be provided when using '--use_norm'.")
    if not config["use_norm"]:
        config["stats_path"] = None

    ds = CharactorMelDataset(
        dataset_dir=config["rootdir"],
        use_norm=config["use_norm"],
        stats_path=config["stats_path"],
        return_guided_attention=False,
        reduction_factor=1,
        n_mels=config["tacotron2_params"]["n_mels"],
        use_fixed_shapes=config["use_fixed_shapes"],
    ).create(batch_size=config["batch_size"], training=True)

    # define model and load weights
    tacotron2 = TFTacotron2(
        config=Tacotron2Config(**config["tacotron2_params"]),
        training=True,  # enable teacher forcing mode
        name="tacotron2",
    )
    tacotron2._build()
    tacotron2.load_weights(args.checkpoint)

    for batch in tqdm(ds, desc="[Extract Duration]"):
        bs, _ = tf.shape(batch["input_ids"])
        # tacotron 2 inference
        *_, alignments = tacotron2(
            batch["input_ids"],
            batch["input_lengths"],
            speaker_ids=tf.zeros([bs]),
            mel_gts=batch["mel_gts"],
            mel_lengths=batch["mel_lengths"],
            use_window_mask=True,
            win_front=config["win_front"],
            win_back=config["win_back"],
            training=True,
        )
        for alignment, char_len, mel_len, utt_id in zip(
            alignments, batch["input_lengths"], batch["mel_lengths"], batch["utt_ids"]
        ):
            alignment = alignment[: char_len - 1, :mel_len]
            idx, _, updates = tf.unique_with_counts(tf.math.argmax(alignment, axis=0))
            duration = tf.scatter_nd(
                tf.expand_dims(idx, -1), updates, [alignment.shape[0]]
            )
            # check that length is compatible
            assert duration.shape[0] == char_len - 1, (
                f"different between len_char and len_durations, {len(duration)} and"
                f" {char_len}"
            )
            assert np.sum(duration) == mel_len, (
                f"different between sum_durations and len_mel, {np.sum(duration)} and"
                f" {mel_len}"
            )
            utt_id_str = utt_id.numpy().decode("utf8")
            np.save(
                os.path.join(config["outdir"], f"{utt_id_str}-durations.npy"),
                duration.numpy().astype(np.int32),
                allow_pickle=False,
            )
            # save alignment to debug
            if config["save_alignment"]:
                figname = os.path.join(config["outdir"], f"{utt_id_str}_alignment.png")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(
                    alignment, aspect="auto", interpolation="none", origin="lower"
                )
                fig.colorbar(im, pad=0.02, aspect=15, orientation="vertical", ax=ax)
                ax.set_title(f"Alignment of {utt_id_str}")
                ax.set_xlabel("Decoder timestep")
                ax.set_ylabel("Encoder timestep")
                plt.tight_layout()
                plt.savefig(figname, bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    main()
