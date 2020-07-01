import os
import re
import numpy as np
import librosa
import soundfile as sf
from pypinyin.style._utils import get_initials, get_finals
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin


_pad = '_'
_eos = '~'
_special = '-'

_initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's',
             'y', 'w']
_finals = ['a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'ong', 'i', 'ia', 'ie', 'iao', 'iou', 'ian',
           'in', 'iang', 'ing', 'iong', 'u', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang', 'ueng', 'un', 'ui',
           'ue', 'iu', 'v', 'vn', 've', 'van']
_r = [i + 'r' for i in _finals]

_tones = ['1', '2', '3', '4', '5']

symbols = [_pad] + [_special] + _initials + [i + j for i in _finals + _r for j in _tones] + [_eos]

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class BakerProcessor(object):

    def __init__(self, root_path, target_rate):
        self.root_path = root_path
        self.target_rate = target_rate

        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin

        items = []
        self.speaker_name = "baker"
        if root_path is not None:
            with open(os.path.join(root_path, 'ProsodyLabeling/000001-010000.txt'), encoding='utf-8') as ttf:
                lines = ttf.readlines()
                for idx in range(0, len(lines), 2):
                    utt_id, text = lines[idx].strip().split()
                    text = lines[idx+1].strip()
                    # text = re.sub(r'#\d', '', text)  # remove prosody, To Do
                    # text = re.sub(r'[。，；“”？#、：！…——]', '', text)
                    # text = pinyin(text, style=Style.TONE3)
                    text = self.get_initials_and_finals(text)
                    wav_path = os.path.join(root_path, 'Wave', '%s.wav' % utt_id)
                    items.append([text, wav_path, self.speaker_name, utt_id])

            self.items = items

    @staticmethod
    def get_initials_and_finals(text):
        result = []
        for x in text.split():
            initials = get_initials(x.strip(), False)
            finals = get_finals(x.strip(), False)
            if initials != "":
                result.append(initials)
            if finals != "":
                result.append(finals)
        return ' '.join(result)

    def get_one_sample(self, idx):
        text, wav_file, speaker_name, utt_id = self.items[idx]

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)
        if rate != self.target_rate:
            assert rate > self.target_rate
            audio = librosa.resample(audio, rate, self.target_rate)

        # convert text to ids
        try:
            text_ids = np.asarray(self.text_to_sequence(text), np.int32)
        except:
            return None
        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": str(int(utt_id)),
            "speaker_name": speaker_name,
            "rate": self.target_rate
        }

        return sample

    @staticmethod
    def text_to_sequence(text):
        global _symbol_to_id

        sequence = []
        for symbol in text.split():
            idx = _symbol_to_id[symbol]
            sequence.append(idx)
        return sequence
