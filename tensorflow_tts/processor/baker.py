import os
import numpy as np
import librosa
import soundfile as sf
from pypinyin.style._utils import get_finals, get_initials
from g2pM import G2pM


_pad = ['_']
_eos = ['~']
_pause = ['sil', 'sp1']
_initials = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
_tones = ['1', '2', '3', '4', '5']
_finals = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie',
           'ii', 'iii', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang',
           'uei', 'uen', 'ueng', 'uo', 'v', 'van', 've', 'vn']
_special = ['io5']

symbols = _pad + _pause + _initials + [i + j for i in _finals for j in _tones] + _special + _eos

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def process_phonelabel(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[12:]
    assert len(lines) % 3 == 0

    text = []
    for i in range(0, len(lines), 3):
        begin = float(lines[i].strip())
        if i == 0:
            assert begin == 0.
        phone = lines[i + 2].strip()
        text.append(phone.replace('"', ''))

    return text


class BakerProcessor(object):

    def __init__(self, data_dir, target_rate=24000, cleaner_names=None):
        self.root_path = data_dir
        self.target_rate = target_rate

        items = []
        self.speaker_name = "baker"
        if data_dir is not None:
            with open(os.path.join(data_dir, 'ProsodyLabeling/000001-010000.txt'), encoding='utf-8') as ttf:
                lines = ttf.readlines()
                for idx in range(0, len(lines), 2):
                    utt_id, _ = lines[idx].strip().split()
                    phonemes = process_phonelabel(os.path.join(data_dir, f'PhoneLabeling/{utt_id}.interval'))
                    phonemes = self.deal_r(phonemes)
                    if 'pl' in phonemes or 'ng1' in phonemes:
                        print(f'Skip this: {utt_id} {phonemes}')
                        continue
                    wav_path = os.path.join(data_dir, 'Wave', '%s.wav' % utt_id)
                    items.append([' '.join(phonemes), wav_path, self.speaker_name, utt_id])
            self.items = items

        self.g2pm = G2pM()

    @staticmethod
    def deal_r(phonemes):
        result = []
        for p in phonemes:
            if p[-1].isdigit() and p[-2] == 'r' and p[:2] != 'er':
                result.append(p[:-2] + p[-1])
                result.append('er5')
            else:
                result.append(p)
        return result

    @staticmethod
    def get_initials_and_finals(text):
        result = []
        for x in text:
            initials = get_initials(x.strip(), False)
            finals = get_finals(x.strip(), False)
            if initials != "":
                # for y and w, we do not have initials
                if initials == 'w' or initials == 'y':
                    pass
                else:
                    result.append(initials)
            if finals != "":
                # we replace ar4 to a4 er5
                if finals[-1].isdigit() and finals[-2] == 'r' and finals[:2] != 'er':
                    result.append(finals[:-2] + finals[-1])
                    result.append('er5')
                else:
                    result.append(finals)
        return ' '.join(result)

    def get_one_sample(self, item):
        text, wav_file, speaker_name, utt_id = item

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)
        if rate != self.target_rate:
            assert rate > self.target_rate
            audio = librosa.resample(audio, rate, self.target_rate)

        # convert text to ids
        try:
            text_ids = np.asarray(self.text_to_sequence(text), np.int32)
        except Exception as e:
            print(e, utt_id, text)
            return None

        # return None
        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": str(int(utt_id)),
            "speaker_name": speaker_name,
            "rate": self.target_rate
        }

        return sample

    def text_to_sequence(self, text, inference=False):
        global _symbol_to_id

        if inference:
            text = self.g2pm(text)
            text = self.get_initials_and_finals(text)
            print(text)

        sequence = []
        for symbol in text.split():
            idx = _symbol_to_id[symbol]
            sequence.append(idx)
        return sequence
