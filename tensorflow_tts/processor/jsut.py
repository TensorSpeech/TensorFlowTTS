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
"""Perform preprocessing and raw feature extraction for JSUT dataset."""

import os
import re

import numpy as np
import soundfile as sf
import pyopenjtalk
import yaml
import librosa
from dataclasses import dataclass
from tensorflow_tts.processor import BaseProcessor
# from tensorflow_tts.utils import cleaners
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME

valid_symbols = [
    'N',
    'a',
    'b',
    'by',
    'ch',
    'cl',
    'd',
    'dy',
    'e',
    'f',
    'g',
    'gy',
    'h',
    'hy',
    'i',
    'j',
    'k',
    'ky',
    'm',
    'my',
    'n',
    'ny',
    'o',
    'p',
    'pau',
    'py',
    'r',
    'ry',
    's',
    'sh',
    't',
    'ts',
    'u',
    'v',
    'w',
    'y',
    'z'
]

_pad = "pad"
_eos = "eos"
_sil = "sil"
# _punctuation = "!'(),.:;? "
# _special = "-"
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
JSUT_SYMBOLS = (
    [_pad] + [_sil] + valid_symbols + [_eos]
)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


@dataclass
class JSUTProcessor(BaseProcessor):
    """JSUT processor."""
    cleaner_names: str = None
    speaker_name: str = "jsut"
    train_f_name: str = "text_kana/basic5000.yaml"

    def create_items(self):
        items = []
        if self.data_dir:
            with open(
                os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                data_json = yaml.load(f, Loader=yaml.FullLoader)

                for k, v in data_json.items():
                    utt_id = k
                    phones = v['phone_level3']
                    phones = phones.split("-")
                    phones = [_sil] + phones + [_sil]
                    wav_path = os.path.join(self.data_dir, "wav", f"{utt_id}.wav")
                    items.append(
                        [" ".join(phones), wav_path, utt_id, self.speaker_name]
                    )
            self.items = items

    def setup_eos_token(self):
        return _eos

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def get_one_sample(self, item):
        text, wav_path, utt_id, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_path)
        audio = audio.astype(np.float32)

        # if rate != self.target_rate:
        #     assert rate > self.target_rate
        #     audio = librosa.resample(audio, rate, self.target_rate)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": utt_id,
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text, inference=False):
        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        if inference:
            text = pyopenjtalk.g2p(text)
            text = text.replace("I", "i")
            text = text.replace("U", "u")
            print(f"phoneme seq: {text}")

        for symbol in text.split():
            idx = self.symbol_to_id[symbol]
            sequence.append(idx)

        # add eos tokens
        sequence += [self.eos_id]
        return sequence
