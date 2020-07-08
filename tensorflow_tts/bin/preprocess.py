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
"""Perform preprocessing, raw feature extraction and train/valid split."""

import argparse
import logging
import os

from pathos.multiprocessing import ProcessingPool as Pool

import librosa
import numpy as np
import yaml
import pyworld as pw

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tensorflow_tts.processor import LJSpeechProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
):
    """Compute log-Mel filterbank feature.
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T))), x_stft


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in tensorflow_tts/bin/preprocess.py)."
    )
    parser.add_argument(
        "--rootdir", default=None, type=str, required=True, help="root path."
    )
    parser.add_argument(
        "--outdir", default=None, type=str, required=True, help="output dir."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=4,
        required=False,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        required=False,
        help="yaml format configuration file.",
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

    processor = LJSpeechProcessor(
        root_path=args.rootdir, cleaner_names="english_cleaners"
    )

    # check directly existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid", "raw-feats"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid", "wavs"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid", "ids"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid", "raw-f0"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "valid", "raw-energies"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train", "raw-feats"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train", "wavs"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train", "ids"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train", "raw-f0"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "train", "raw-energies"), exist_ok=True)

    # train test split
    idx_train, idx_valid = train_test_split(
        range(len(processor.items)),
        shuffle=True,
        test_size=args.test_size,
        random_state=42,
    )

    # train/valid utt_ids
    train_utt_ids = []
    valid_utt_ids = []

    for idx in range(len(processor.items)):
        utt_ids = processor.get_one_sample(idx)["utt_id"]
        if idx in idx_train:
            train_utt_ids.append(utt_ids)
        elif idx in idx_valid:
            valid_utt_ids.append(utt_ids)

    # save train and valid utt_ids to track later.
    np.save(os.path.join(args.outdir, "train_utt_ids.npy"), train_utt_ids)
    np.save(os.path.join(args.outdir, "valid_utt_ids.npy"), valid_utt_ids)

    pbar = tqdm(initial=0, total=len(processor.items), desc="[Preprocessing]")

    # process each data
    def save_to_file(idx):
        sample = processor.get_one_sample(idx)

        # get info from sample.
        audio = sample["audio"]
        text_ids = sample["text_ids"]
        utt_id = sample["utt_id"]
        rate = sample["rate"]

        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        assert (
            rate == config["sampling_rate"]
        ), f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]
        else:
            x = librosa.resample(audio, rate, config["sampling_rate_for_feats"])
            sampling_rate = config["sampling_rate_for_feats"]
            assert (
                config["hop_size"] * config["sampling_rate_for_feats"] % rate == 0
            ), "hop_size must be int value. please check sampling_rate_for_feats is correct."
            hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // rate

        # extract feature
        mel, x_stft = logmelfilterbank(
            x,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=config["window"],
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
        )

        # make sure the audio length and feature length
        audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
        audio = audio[: len(mel) * config["hop_size"]]

        # extract raw pitch
        f0, _ = pw.dio(
            x.astype(np.double),
            fs=config["sampling_rate"],
            f0_ceil=config["fmax"],
            frame_period=1000 * config["hop_size"] / config["sampling_rate"],
        )

        if len(f0) >= len(mel):
            f0 = f0[: len(mel)]
        else:
            f0 = np.pad(f0, ((0, len(mel) - len(f0))))

        # extract energy
        S = librosa.magphase(x_stft)[0]
        energy = np.sqrt(np.sum(S ** 2, axis=0))

        assert len(mel) * config["hop_size"] == len(audio)
        assert len(mel) == len(f0) == len(energy)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(
                f"{utt_id} causes clipping. "
                f"it is better to re-consider global gain scale."
            )

        # save
        if config["format"] == "npy":
            if idx in idx_train:
                subdir = "train"
            elif idx in idx_valid:
                subdir = "valid"

            np.save(
                os.path.join(args.outdir, subdir, "wavs", f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(
                    args.outdir, subdir, "raw-feats", f"{utt_id}-raw-feats.npy"
                ),
                mel.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.outdir, subdir, "ids", f"{utt_id}-ids.npy"),
                text_ids.astype(np.int32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.outdir, subdir, "raw-f0", f"{utt_id}-raw-f0.npy"),
                f0.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(
                    args.outdir, subdir, "raw-energies", f"{utt_id}-raw-energy.npy"
                ),
                energy.astype(np.float32),
                allow_pickle=False,
            )
        else:
            raise ValueError("support only npy format.")

        pbar.update(1)

    # apply multi-processing Pool
    p = Pool(nodes=args.n_cpus)
    p.map(save_to_file, range(len(processor.items)))
    pbar.close()


if __name__ == "__main__":
    main()
