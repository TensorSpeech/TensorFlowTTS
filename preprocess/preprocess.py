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
import random
import argparse
import logging
import os
from pathlib import Path
import librosa
import numpy as np
import pyworld as pw
import yaml
from pathos.multiprocessing import ProcessingPool as Pool
import shutil
from tqdm import tqdm
from experiment.example_dataset import LJSpeechProcessor
from experiment.example_dataset import _symbol_to_id

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
        default=8,
        required=False,
        help="number of CPUs to use for multi-processing.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        required=False,
        help="the proportion of the dataset to include in the test split. (default=0.05)",
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
        root_path=args.rootdir, speakers_map={"": 0}  # TODO FIX
    )

    out_dirs = ["raw-feats", "wavs", "ids", "raw-f0", "raw-energies"]
    [Path(Path(args.outdir).joinpath(f"train/{i}")).mkdir(parents=True, exist_ok=True) for i in out_dirs]
    [Path(Path(args.outdir).joinpath(f"valid/{i}")).mkdir(parents=True, exist_ok=True) for i in out_dirs]

    # # train test split


    def ph_based_trim(utt_id: str, text_ids: np.array, raw_text: str, audio: np.array, hop_size: int):

        duration_path = config.get("duration_path", "dataset/durations")
        duration_fixed_path = config.get("duration_fixed_path", "dataset/trimmed-durations")
        sil_ph = ["SIL", "END"]  # TODO FIX hardcoded values
        text = raw_text.split(" ")

        trim_start, trim_end = False, False

        if text[0] in sil_ph:
            trim_start = True

        if text[-1] in sil_ph:
            trim_end = True

        if not trim_start and not trim_end:
            return False, text_ids, audio

        idx_start, idx_end = 0 if not trim_start else 1, text_ids.__len__() if not trim_end else -1
        text_ids = text_ids[idx_start:idx_end]
        durations = np.load(f"{duration_path}/{utt_id}-durations.npy")
        if trim_start:
            s_trim = int(durations[0] * hop_size)
            audio = audio[s_trim:]
        if trim_end:
            e_trim = int(durations[-1] * hop_size)
            audio = audio[:-e_trim]

        durations = durations[idx_start:idx_end]
        np.save(f"{duration_fixed_path}/{utt_id}-durations.npy", durations)
        return True, text_ids, audio


    # process each data
    def save_to_file(sample):
        # sample = processor.get_one_sample(idx)

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
            if config.get("ph_based_trim", True):  # TODO FIX
                _, text_ids, audio = ph_based_trim(utt_id, text_ids,
                                                   sample["raw_text"], audio, config["hop_size"])
            else:
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
            # if idx in idx_train: # TODO FIX
            subdir = "train"
            # elif idx in idx_valid:
            #     subdir = "valid"

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

    before_split = {}  # {speaker_name: [utt_ids, ...]}
    max_samples = 100  # TODO FIX LATER
    samples = []

    p = Pool(nodes=args.n_cpus)
    p_bar = tqdm(total=len(processor.items))

    for idx in range(len(processor.items)):
        sample = processor.get_one_sample(idx)
        if sample["speaker_name"] not in before_split:
            before_split[sample["speaker_name"]] = [sample["utt_id"]]
        before_split[sample["speaker_name"]].append(sample["utt_id"])
        samples.append(sample)
        if len(samples) >= max_samples:
            for _ in p.imap(save_to_file, samples):
                p_bar.update(1)
            samples = []

    train_utt_ids = []
    valid_utt_ids = []
    val_size = args.test_size
    for speaker in before_split.keys():

        random.shuffle(before_split[speaker])
        examples = before_split[speaker]

        split_idx = int(len(examples) * val_size)

        train_utt_ids.extend(examples[split_idx:])
        valid_utt_ids.extend(examples[:split_idx])


    np.save(os.path.join(args.outdir, "train_utt_ids.npy"), train_utt_ids)
    np.save(os.path.join(args.outdir, "valid_utt_ids.npy"), valid_utt_ids)

    train_path = os.path.join(args.outdir, "train")
    val_path = os.path.join(args.outdir, "valid")
    for f_path in os.listdir(train_path):
        for k in os.listdir(f"{train_path}/{f_path}"):
            if k.split("-")[0] in valid_utt_ids:
                shutil.move(f"{train_path}/{f_path}/{k}", f"{val_path}/{f_path}/{k}")


if __name__ == "__main__":
    main()
