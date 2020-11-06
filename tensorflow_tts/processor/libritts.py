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
"""Perform preprocessing and raw feature extraction for LibriTTS dataset."""

import os
import re

import numpy as np
import soundfile as sf
from dataclasses import dataclass

from g2p_en import g2p as grapheme_to_phonem

from tensorflow_tts.processor.base_processor import BaseProcessor

g2p = grapheme_to_phonem.G2p()

valid_symbols = g2p.phonemes
valid_symbols.append("SIL")
valid_symbols.append("END")

_punctuation = "!'(),.:;? "
_arpabet = ["@" + s for s in valid_symbols]

LIBRITTS_SYMBOLS = _arpabet + list(_punctuation)


@dataclass
class LibriTTSProcessor(BaseProcessor):

    mode: str = "train"
    train_f_name: str = "train.txt"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    cleaner_names: str = None

    def create_items(self):
        with open(
            os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8"
        ) as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension
                    if wav_path[-len(self.f_extension) :] != self.f_extension
                    else wav_path
                )
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item
        audio, rate = sf.read(wav_path, dtype="float32")

        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": wav_path.split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return None # because we do not use this 

    def text_to_sequence(self, text):
        if (
            self.mode == "train"
        ):  # in train mode text should be already transformed to phonemes
            return self.symbols_to_ids(self.clean_g2p(text.split(" ")))
        else:
            return self.inference_text_to_seq(text)

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

    def text_to_ph(self, text: str):
        return self.clean_g2p(g2p(text))

    def clean_g2p(self, g2p_text: list):
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
            if txt != " ":
                data.append("@" + txt) 
        return data
