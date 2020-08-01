# -*- coding: utf-8 -*-
# This code is copy and modify from https://github.com/keithito/tacotron.
"""Perform preprocessing and raw feature extraction."""

import os
import re

import numpy as np
import soundfile as sf
from g2p_en import G2p

from tensorflow_tts.utils import cleaners

valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",#Silence
    "END", #padding token
]

_pad = "_"
_eos = "~"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_singlesil = [",",";","?",".","..."]
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
symbols = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + [_eos]
)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


class LJSpeechProcessor(object):
    """LJSpeech processor."""

    def __init__(self, data_dir, cleaner_names, metadata_filename="metadata.csv"):
        self.data_dir = data_dir
        self.cleaner_names = cleaner_names

        self.speaker_name = "ljspeech"
        if data_dir:
            with open(os.path.join(data_dir, metadata_filename), encoding="utf-8") as f:
                self.items = [self.split_line(data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        wav_file, _, text_norm = line.strip().split(split)
        wav_path = os.path.join(data_dir, "wavs", f"{wav_file}.wav")
        return text_norm, wav_path, self.speaker_name

    def get_one_sample(self, item):
        text, wav_file, speaker_name = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": os.path.split(wav_file)[-1].split(".")[0],
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
                _clean_text(m.group(1), [self.cleaner_names])
            )
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
        return sequence
      
    def processtxtph(self,intxt):
      g2p = G2p()
      ptext =  _clean_text(intxt,[self.cleaner_names])
      phs = _g2p2synth(g2p(ptext))

      arpatxt = " ".join(phs)
      ids = _arpabet_to_sequence(arpatxt)

      return ids, arpatxt
  
  
  


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text

def _g2p2synth(inseq):
  phseq = list()
  for idx, itm in enumerate(inseq):
    if itm == ' ':
      continue
    
    if idx < len(inseq) - 1: #Prevent it from appending SILs due to end periods
      if itm in _singlesil:
       phseq.append("SIL")
       continue
    else:
      if itm in _singlesil:
        continue # Skip ending dots

    phseq.append(itm)
    
  phseq.append("SIL")
  return phseq




def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
