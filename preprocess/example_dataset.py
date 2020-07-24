from dataclasses import dataclass

import numpy as np
import soundfile as sf
from g2p_en import g2p as grapheme_to_phonem

from base_dataset import BaseDataset

g2p = grapheme_to_phonem.G2p()

valid_symbols = g2p.phonemes
valid_symbols.append("SIL")
valid_symbols.append("END")

_punctuation = "!'(),.:;? "
_arpabet = ["@" + s for s in valid_symbols]

symbols = (_arpabet + list(_punctuation))

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


@dataclass
class LJSpeechProcessor(BaseDataset):

    mode: str = "train"

    def get_one_sample(self, idx: int):
        text, wav_file, speaker_name = self.items[idx]
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

    def text_to_sequence(self, text: str):
        if self.mode == "train":  # in train mode text should be already transformed to phonemes
            return symbols_to_ids(clean_g2p(text.split(" ")))
        else:
            return self.inference_text_to_seq(text)

    @staticmethod
    def inference_text_to_seq(text: str):
        symbols_to_ids(text_to_ph(text))


def text_to_ph(text: str):
    return clean_g2p(g2p(text))


def clean_g2p(g2p_text: list):
    data = []
    for i, txt in enumerate(g2p_text):
        if i == len(g2p_text) - 1:
            if txt != " ":
                data.append("@" + txt)
            data.append("@END")  # TODO try learning without end token and compare results
            break
        data.append("@" + txt) if txt != " " else "@SIL"
    return data


def symbols_to_ids(symbols_list: list):
    return [_symbol_to_id[s] for s in symbols_list]

