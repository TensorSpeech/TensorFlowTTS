# -*- coding: utf-8 -*-
# Copyright 2020 The TensorFlowTTS Team
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
"""Example Processor."""

from dataclasses import dataclass

import numpy as np
import soundfile as sf
from g2p_en import g2p as grapheme_to_phonem

from .base_processor import BaseProcessor

g2p = grapheme_to_phonem.G2p()

valid_symbols = g2p.phonemes
valid_symbols.append("SIL")
valid_symbols.append("END")

_punctuation = "!'(),.:;? "
_arpabet = ["@" + s for s in valid_symbols]

symbols = _arpabet + list(_punctuation)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


@dataclass
class ExampleMultispeaker(BaseProcessor):

    mode: str = "train"

    def get_one_sample(self, item):

        text, wav_file, speaker_name = item
        audio, rate = sf.read(wav_file, dtype="float32")

        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": wav_file.split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text: str):
        if (
            self.mode == "train"
        ):  # in train mode text should be already transformed to phonemes
            return symbols_to_ids(clean_g2p(text.split(" ")))
        else:
            return self.inference_text_to_seq(text)

    @staticmethod
    def inference_text_to_seq(text: str):
        return symbols_to_ids(text_to_ph(text))


def text_to_ph(text: str):
    return clean_g2p(g2p(text))


def clean_g2p(g2p_text: list):
    data = []
    for i, txt in enumerate(g2p_text):
        if i == len(g2p_text) - 1:
            if txt != " " and txt != "SIL":
                data.append("@" + txt)
            else:
                data.append(
                    "@END"
                )  # TODO try learning without end token and compare results
            break
        data.append("@" + txt) if txt != " " else data.append(
            "@SIL"
        )  # TODO change it in inference
    return data


def symbols_to_ids(symbols_list: list):
    return [_symbol_to_id[s] for s in symbols_list]
