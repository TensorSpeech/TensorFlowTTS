# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
"""Fix mismatch between sum durations and mel lengths."""

import numpy as np
import os
from tqdm import tqdm
import click
import logging
import sys


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@click.command()
@click.option("--base_path", default="dump")
@click.option("--trimmed_dur_path", default="dataset/trimmed-durations")
@click.option("--dur_path", default="dataset/durations")
@click.option("--use_norm", default="f")
def fix(base_path: str, dur_path: str, trimmed_dur_path: str, use_norm: str):
    for t in ["train", "valid"]:
        mfa_longer = []
        mfa_shorter = []
        big_diff = []
        not_fixed = []
        pre_path = os.path.join(base_path, t)
        os.makedirs(os.path.join(pre_path, "fix_dur"), exist_ok=True)

        logging.info(f"FIXING {t} set ...\n")
        for i in tqdm(os.listdir(os.path.join(pre_path, "ids"))):
            if use_norm == "t":
                mel = np.load(
                    os.path.join(
                        pre_path, "norm-feats", f"{i.split('-')[0]}-norm-feats.npy"
                    )
                )
            else:
                mel = np.load(
                    os.path.join(
                        pre_path, "raw-feats", f"{i.split('-')[0]}-raw-feats.npy"
                    )
                )

            try:
                dur = np.load(
                    os.path.join(trimmed_dur_path, f"{i.split('-')[0]}-durations.npy")
                )
            except:
                dur = np.load(
                    os.path.join(dur_path, f"{i.split('-')[0]}-durations.npy")
                )

            l_mel = len(mel)
            dur_s = np.sum(dur)
            cloned = np.array(dur, copy=True)
            diff = abs(l_mel - dur_s)

            if abs(l_mel - dur_s) > 30:  # more then 300 ms
                big_diff.append([i, abs(l_mel - dur_s)])

            if dur_s > l_mel:
                for j in range(1, len(dur) - 1):
                    if diff == 0:
                        break
                    dur_val = cloned[-j]

                    if dur_val >= diff:
                        cloned[-j] -= diff
                        diff -= dur_val
                        break
                    else:
                        cloned[-j] = 0
                        diff -= dur_val

                    if j == len(dur) - 2:
                        not_fixed.append(i)

                mfa_longer.append(abs(l_mel - dur_s))
            elif dur_s < l_mel:
                cloned[-1] += diff
                mfa_shorter.append(abs(l_mel - dur_s))

            np.save(
                os.path.join(pre_path, "fix_dur", f"{i.split('-')[0]}-durations.npy"),
                cloned.astype(np.int32),
                allow_pickle=False,
            )

        logging.info(
            f"{t} stats: number of mfa with longer duration: {len(mfa_longer)}, total diff: {sum(mfa_longer)}"
            f", mean diff: {sum(mfa_longer)/len(mfa_longer) if len(mfa_longer) > 0 else 0}"
        )
        logging.info(
            f"{t} stats: number of mfa with shorter duration: {len(mfa_shorter)}, total diff: {sum(mfa_shorter)}"
            f", mean diff: {sum(mfa_shorter)/len(mfa_shorter) if len(mfa_shorter) > 0 else 0}"
        )
        logging.info(
            f"{t} stats: number of files with a ''big'' duration diff: {len(big_diff)} if number>1 you should check it"
        )
        logging.info(f"{t} stats: not fixed len: {len(not_fixed)}\n")


if __name__ == "__main__":
    fix()
