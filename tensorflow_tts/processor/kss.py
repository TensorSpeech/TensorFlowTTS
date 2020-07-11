# -*- coding: utf-8 -*-
# This code is copy and modify from https://github.com/keithito/tacotron.
"""Perform preprocessing and raw feature extraction."""

import re
import os

import numpy as np
import soundfile as sf

from tensorflow_tts.utils import cleaners

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""

from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from tensorflow_tts.utils.korean import ALL_SYMBOLS, PAD, EOS

# For english
en_symbols = (
    PAD + EOS + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "
)  # <-For deployment(Because korean ALL_SYMBOLS follow this convention)

symbols = ALL_SYMBOLS  # for korean

"""
초성과 종성은 같아보이지만, 다른 character이다.

'_~ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ!'(),-.:;? '

'_': 0, '~': 1, 'ᄀ': 2, 'ᄁ': 3, 'ᄂ': 4, 'ᄃ': 5, 'ᄄ': 6, 'ᄅ': 7, 'ᄆ': 8, 'ᄇ': 9, 'ᄈ': 10, 
'ᄉ': 11, 'ᄊ': 12, 'ᄋ': 13, 'ᄌ': 14, 'ᄍ': 15, 'ᄎ': 16, 'ᄏ': 17, 'ᄐ': 18, 'ᄑ': 19, 'ᄒ': 20, 
'ᅡ': 21, 'ᅢ': 22, 'ᅣ': 23, 'ᅤ': 24, 'ᅥ': 25, 'ᅦ': 26, 'ᅧ': 27, 'ᅨ': 28, 'ᅩ': 29, 'ᅪ': 30, 
'ᅫ': 31, 'ᅬ': 32, 'ᅭ': 33, 'ᅮ': 34, 'ᅯ': 35, 'ᅰ': 36, 'ᅱ': 37, 'ᅲ': 38, 'ᅳ': 39, 'ᅴ': 40, 
'ᅵ': 41, 'ᆨ': 42, 'ᆩ': 43, 'ᆪ': 44, 'ᆫ': 45, 'ᆬ': 46, 'ᆭ': 47, 'ᆮ': 48, 'ᆯ': 49, 'ᆰ': 50, 
'ᆱ': 51, 'ᆲ': 52, 'ᆳ': 53, 'ᆴ': 54, 'ᆵ': 55, 'ᆶ': 56, 'ᆷ': 57, 'ᆸ': 58, 'ᆹ': 59, 'ᆺ': 60, 
'ᆻ': 61, 'ᆼ': 62, 'ᆽ': 63, 'ᆾ': 64, 'ᆿ': 65, 'ᇀ': 66, 'ᇁ': 67, 'ᇂ': 68, '!': 69, "'": 70, 
'(': 71, ')': 72, ',': 73, '-': 74, '.': 75, ':': 76, ';': 77, '?': 78, ' ': 79
"""

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


class KSSProcessor(object):
    """KSS processor."""

    def __init__(self, root_path, cleaner_names):
        self.root_path = root_path
        self.cleaner_names = cleaner_names

        items = []
        self.speaker_name = "kss"
        if root_path is not None:
            with open(os.path.join(root_path, "transcript.v.1.2.txt"), encoding="utf-8") as ttf:
                for line in ttf:
                    parts = line.strip().split("|")
                    wav_path = os.path.join(root_path, parts[0])
                    text = parts[2]
                    items.append([text, wav_path, self.speaker_name])

            self.items = items

    def get_one_sample(self, idx):
        text, wav_file, speaker_name = self.items[idx]

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": self.items[idx][1].split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        global _symbol_to_id

        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence(
                    _clean_text(text, [self.cleaner_names])
                )
                break
            sequence += _symbols_to_sequence(
                _clean_text(m.group(1), self.cleaner_names)
            )
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
