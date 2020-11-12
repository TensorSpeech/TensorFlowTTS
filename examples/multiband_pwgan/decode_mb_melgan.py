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
"""Decode trained Mb-Melgan from folder."""

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.datasets import MelDataset
from tensorflow_tts.models import TFPQMF, TFMelGANGenerator


def main():
    """Run melgan decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Generate Audio from melspectrogram with trained melgan "
        "(See detail in example/melgan/decode_melgan.py)."
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
        "--use-norm", type=int, default=1, help="Use norm or raw melspectrogram."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch_size.")
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
        mel_query = "*-fs-after-feats.npy" if "fastspeech" in args.rootdir else "*-norm-feats.npy" if args.use_norm == 1 else "*-raw-feats.npy"
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = MelDataset(
        root_dir=args.rootdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
    )
    dataset = dataset.create(batch_size=args.batch_size)

    # define model and load checkpoint
    mb_melgan = TFMelGANGenerator(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator"]),
        name="multiband_melgan_generator",
    )
    mb_melgan._build()
    mb_melgan.load_weights(args.checkpoint)

    pqmf = TFPQMF(
        config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator"]), name="pqmf"
    )

    for data in tqdm(dataset, desc="[Decoding]"):
        utt_ids, mels, mel_lengths = data["utt_ids"], data["mels"], data["mel_lengths"]

        # melgan inference.
        generated_subbands = mb_melgan(mels)
        generated_audios = pqmf.synthesis(generated_subbands)

        # convert to numpy.
        generated_audios = generated_audios.numpy()  # [B, T]

        # save to outdir
        for i, audio in enumerate(generated_audios):
            utt_id = utt_ids[i].numpy().decode("utf-8")
            sf.write(
                os.path.join(args.outdir, f"{utt_id}.wav"),
                audio[: mel_lengths[i].numpy() * config["hop_size"]],
                config["sampling_rate"],
                "PCM_16",
            )


if __name__ == "__main__":
    main()
