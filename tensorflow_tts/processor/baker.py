import os
import numpy as np
import librosa
import soundfile as sf
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin


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



pinyin_dict = {
    'a': ('', 'a'),
    'ai': ('', 'ai'),
    'an': ('', 'an'),
    'ang': ('', 'ang'),
    'ao': ('', 'ao'),
    'ba': ('b', 'a'),
    'bai': ('b', 'ai'),
    'ban': ('b', 'an'),
    'bang': ('b', 'ang'),
    'bao': ('b', 'ao'),
    'be': ('b', 'e'),
    'bei': ('b', 'ei'),
    'ben': ('b', 'en'),
    'beng': ('b', 'eng'),
    'bi': ('b', 'i'),
    'bian': ('b', 'ian'),
    'biao': ('b', 'iao'),
    'bie': ('b', 'ie'),
    'bin': ('b', 'in'),
    'bing': ('b', 'ing'),
    'bo': ('b', 'o'),
    'bu': ('b', 'u'),
    'ca': ('c', 'a'),
    'cai': ('c', 'ai'),
    'can': ('c', 'an'),
    'cang': ('c', 'ang'),
    'cao': ('c', 'ao'),
    'ce': ('c', 'e'),
    'cen': ('c', 'en'),
    'ceng': ('c', 'eng'),
    'cha': ('ch', 'a'),
    'chai': ('ch', 'ai'),
    'chan': ('ch', 'an'),
    'chang': ('ch', 'ang'),
    'chao': ('ch', 'ao'),
    'che': ('ch', 'e'),
    'chen': ('ch', 'en'),
    'cheng': ('ch', 'eng'),
    'chi': ('ch', 'iii'),
    'chong': ('ch', 'ong'),
    'chou': ('ch', 'ou'),
    'chu': ('ch', 'u'),
    'chua': ('ch', 'ua'),
    'chuai': ('ch', 'uai'),
    'chuan': ('ch', 'uan'),
    'chuang': ('ch', 'uang'),
    'chui': ('ch', 'uei'),
    'chun': ('ch', 'uen'),
    'chuo': ('ch', 'uo'),
    'ci': ('c', 'ii'),
    'cong': ('c', 'ong'),
    'cou': ('c', 'ou'),
    'cu': ('c', 'u'),
    'cuan': ('c', 'uan'),
    'cui': ('c', 'uei'),
    'cun': ('c', 'uen'),
    'cuo': ('c', 'uo'),
    'da': ('d', 'a'),
    'dai': ('d', 'ai'),
    'dan': ('d', 'an'),
    'dang': ('d', 'ang'),
    'dao': ('d', 'ao'),
    'de': ('d', 'e'),
    'dei': ('d', 'ei'),
    'den': ('d', 'en'),
    'deng': ('d', 'eng'),
    'di': ('d', 'i'),
    'dia': ('d', 'ia'),
    'dian': ('d', 'ian'),
    'diao': ('d', 'iao'),
    'die': ('d', 'ie'),
    'ding': ('d', 'ing'),
    'diu': ('d', 'iou'),
    'dong': ('d', 'ong'),
    'dou': ('d', 'ou'),
    'du': ('d', 'u'),
    'duan': ('d', 'uan'),
    'dui': ('d', 'uei'),
    'dun': ('d', 'uen'),
    'duo': ('d', 'uo'),
    'e': ('', 'e'),
    'ei': ('', 'ei'),
    'en': ('', 'en'),
    'ng': ('', 'en'),
    'eng': ('', 'eng'),
    'er': ('', 'er'),
    'fa': ('f', 'a'),
    'fan': ('f', 'an'),
    'fang': ('f', 'ang'),
    'fei': ('f', 'ei'),
    'fen': ('f', 'en'),
    'feng': ('f', 'eng'),
    'fo': ('f', 'o'),
    'fou': ('f', 'ou'),
    'fu': ('f', 'u'),
    'ga': ('g', 'a'),
    'gai': ('g', 'ai'),
    'gan': ('g', 'an'),
    'gang': ('g', 'ang'),
    'gao': ('g', 'ao'),
    'ge': ('g', 'e'),
    'gei': ('g', 'ei'),
    'gen': ('g', 'en'),
    'geng': ('g', 'eng'),
    'gong': ('g', 'ong'),
    'gou': ('g', 'ou'),
    'gu': ('g', 'u'),
    'gua': ('g', 'ua'),
    'guai': ('g', 'uai'),
    'guan': ('g', 'uan'),
    'guang': ('g', 'uang'),
    'gui': ('g', 'uei'),
    'gun': ('g', 'uen'),
    'guo': ('g', 'uo'),
    'ha': ('h', 'a'),
    'hai': ('h', 'ai'),
    'han': ('h', 'an'),
    'hang': ('h', 'ang'),
    'hao': ('h', 'ao'),
    'he': ('h', 'e'),
    'hei': ('h', 'ei'),
    'hen': ('h', 'en'),
    'heng': ('h', 'eng'),
    'hong': ('h', 'ong'),
    'hou': ('h', 'ou'),
    'hu': ('h', 'u'),
    'hua': ('h', 'ua'),
    'huai': ('h', 'uai'),
    'huan': ('h', 'uan'),
    'huang': ('h', 'uang'),
    'hui': ('h', 'uei'),
    'hun': ('h', 'uen'),
    'huo': ('h', 'uo'),
    'ji': ('j', 'i'),
    'jia': ('j', 'ia'),
    'jian': ('j', 'ian'),
    'jiang': ('j', 'iang'),
    'jiao': ('j', 'iao'),
    'jie': ('j', 'ie'),
    'jin': ('j', 'in'),
    'jing': ('j', 'ing'),
    'jiong': ('j', 'iong'),
    'jiu': ('j', 'iou'),
    'ju': ('j', 'v'),
    'juan': ('j', 'van'),
    'jue': ('j', 've'),
    'jun': ('j', 'vn'),
    'ka': ('k', 'a'),
    'kai': ('k', 'ai'),
    'kan': ('k', 'an'),
    'kang': ('k', 'ang'),
    'kao': ('k', 'ao'),
    'ke': ('k', 'e'),
    'kei': ('k', 'ei'),
    'ken': ('k', 'en'),
    'keng': ('k', 'eng'),
    'kong': ('k', 'ong'),
    'kou': ('k', 'ou'),
    'ku': ('k', 'u'),
    'kua': ('k', 'ua'),
    'kuai': ('k', 'uai'),
    'kuan': ('k', 'uan'),
    'kuang': ('k', 'uang'),
    'kui': ('k', 'uei'),
    'kun': ('k', 'uen'),
    'kuo': ('k', 'uo'),
    'la': ('l', 'a'),
    'lai': ('l', 'ai'),
    'lan': ('l', 'an'),
    'lang': ('l', 'ang'),
    'lao': ('l', 'ao'),
    'le': ('l', 'e'),
    'lei': ('l', 'ei'),
    'leng': ('l', 'eng'),
    'li': ('l', 'i'),
    'lia': ('l', 'ia'),
    'lian': ('l', 'ian'),
    'liang': ('l', 'iang'),
    'liao': ('l', 'iao'),
    'lie': ('l', 'ie'),
    'lin': ('l', 'in'),
    'ling': ('l', 'ing'),
    'liu': ('l', 'iou'),
    'lo': ('l', 'o'),
    'long': ('l', 'ong'),
    'lou': ('l', 'ou'),
    'lu': ('l', 'u'),
    'lv': ('l', 'v'),
    'luan': ('l', 'uan'),
    'lve': ('l', 've'),
    'lue': ('l', 've'),
    'lun': ('l', 'uen'),
    'luo': ('l', 'uo'),
    'ma': ('m', 'a'),
    'mai': ('m', 'ai'),
    'man': ('m', 'an'),
    'mang': ('m', 'ang'),
    'mao': ('m', 'ao'),
    'me': ('m', 'e'),
    'mei': ('m', 'ei'),
    'men': ('m', 'en'),
    'meng': ('m', 'eng'),
    'mi': ('m', 'i'),
    'mian': ('m', 'ian'),
    'miao': ('m', 'iao'),
    'mie': ('m', 'ie'),
    'min': ('m', 'in'),
    'ming': ('m', 'ing'),
    'miu': ('m', 'iou'),
    'mo': ('m', 'o'),
    'mou': ('m', 'ou'),
    'mu': ('m', 'u'),
    'na': ('n', 'a'),
    'nai': ('n', 'ai'),
    'nan': ('n', 'an'),
    'nang': ('n', 'ang'),
    'nao': ('n', 'ao'),
    'ne': ('n', 'e'),
    'nei': ('n', 'ei'),
    'nen': ('n', 'en'),
    'neng': ('n', 'eng'),
    'ni': ('n', 'i'),
    'nia': ('n', 'ia'),
    'nian': ('n', 'ian'),
    'niang': ('n', 'iang'),
    'niao': ('n', 'iao'),
    'nie': ('n', 'ie'),
    'nin': ('n', 'in'),
    'ning': ('n', 'ing'),
    'niu': ('n', 'iou'),
    'nong': ('n', 'ong'),
    'nou': ('n', 'ou'),
    'nu': ('n', 'u'),
    'nv': ('n', 'v'),
    'nuan': ('n', 'uan'),
    'nve': ('n', 've'),
    'nue': ('n', 've'),
    'nuo': ('n', 'uo'),
    'o': ('', 'o'),
    'ou': ('', 'ou'),
    'pa': ('p', 'a'),
    'pai': ('p', 'ai'),
    'pan': ('p', 'an'),
    'pang': ('p', 'ang'),
    'pao': ('p', 'ao'),
    'pe': ('p', 'e'),
    'pei': ('p', 'ei'),
    'pen': ('p', 'en'),
    'peng': ('p', 'eng'),
    'pi': ('p', 'i'),
    'pian': ('p', 'ian'),
    'piao': ('p', 'iao'),
    'pie': ('p', 'ie'),
    'pin': ('p', 'in'),
    'ping': ('p', 'ing'),
    'po': ('p', 'o'),
    'pou': ('p', 'ou'),
    'pu': ('p', 'u'),
    'qi': ('q', 'i'),
    'qia': ('q', 'ia'),
    'qian': ('q', 'ian'),
    'qiang': ('q', 'iang'),
    'qiao': ('q', 'iao'),
    'qie': ('q', 'ie'),
    'qin': ('q', 'in'),
    'qing': ('q', 'ing'),
    'qiong': ('q', 'iong'),
    'qiu': ('q', 'iou'),
    'qu': ('q', 'v'),
    'quan': ('q', 'van'),
    'que': ('q', 've'),
    'qun': ('q', 'vn'),
    'ran': ('r', 'an'),
    'rang': ('r', 'ang'),
    'rao': ('r', 'ao'),
    're': ('r', 'e'),
    'ren': ('r', 'en'),
    'reng': ('r', 'eng'),
    'ri': ('r', 'iii'),
    'rong': ('r', 'ong'),
    'rou': ('r', 'ou'),
    'ru': ('r', 'u'),
    'rua': ('r', 'ua'),
    'ruan': ('r', 'uan'),
    'rui': ('r', 'uei'),
    'run': ('r', 'uen'),
    'ruo': ('r', 'uo'),
    'sa': ('s', 'a'),
    'sai': ('s', 'ai'),
    'san': ('s', 'an'),
    'sang': ('s', 'ang'),
    'sao': ('s', 'ao'),
    'se': ('s', 'e'),
    'sen': ('s', 'en'),
    'seng': ('s', 'eng'),
    'sha': ('sh', 'a'),
    'shai': ('sh', 'ai'),
    'shan': ('sh', 'an'),
    'shang': ('sh', 'ang'),
    'shao': ('sh', 'ao'),
    'she': ('sh', 'e'),
    'shei': ('sh', 'ei'),
    'shen': ('sh', 'en'),
    'sheng': ('sh', 'eng'),
    'shi': ('sh', 'iii'),
    'shou': ('sh', 'ou'),
    'shu': ('sh', 'u'),
    'shua': ('sh', 'ua'),
    'shuai': ('sh', 'uai'),
    'shuan': ('sh', 'uan'),
    'shuang': ('sh', 'uang'),
    'shui': ('sh', 'uei'),
    'shun': ('sh', 'uen'),
    'shuo': ('sh', 'uo'),
    'si': ('s', 'ii'),
    'song': ('s', 'ong'),
    'sou': ('s', 'ou'),
    'su': ('s', 'u'),
    'suan': ('s', 'uan'),
    'sui': ('s', 'uei'),
    'sun': ('s', 'uen'),
    'suo': ('s', 'uo'),
    'ta': ('t', 'a'),
    'tai': ('t', 'ai'),
    'tan': ('t', 'an'),
    'tang': ('t', 'ang'),
    'tao': ('t', 'ao'),
    'te': ('t', 'e'),
    'tei': ('t', 'ei'),
    'teng': ('t', 'eng'),
    'ti': ('t', 'i'),
    'tian': ('t', 'ian'),
    'tiao': ('t', 'iao'),
    'tie': ('t', 'ie'),
    'ting': ('t', 'ing'),
    'tong': ('t', 'ong'),
    'tou': ('t', 'ou'),
    'tu': ('t', 'u'),
    'tuan': ('t', 'uan'),
    'tui': ('t', 'uei'),
    'tun': ('t', 'uen'),
    'tuo': ('t', 'uo'),
    'wa': ('', 'ua'),
    'wai': ('', 'uai'),
    'wan': ('', 'uan'),
    'wang': ('', 'uang'),
    'wei': ('', 'uei'),
    'wen': ('', 'uen'),
    'weng': ('', 'ueng'),
    'wo': ('', 'uo'),
    'wu': ('', 'u'),
    'xi': ('x', 'i'),
    'xia': ('x', 'ia'),
    'xian': ('x', 'ian'),
    'xiang': ('x', 'iang'),
    'xiao': ('x', 'iao'),
    'xie': ('x', 'ie'),
    'xin': ('x', 'in'),
    'xing': ('x', 'ing'),
    'xiong': ('x', 'iong'),
    'xiu': ('x', 'iou'),
    'xu': ('x', 'v'),
    'xuan': ('x', 'van'),
    'xue': ('x', 've'),
    'xun': ('x', 'vn'),
    'ya': ('', 'ia'),
    'yan': ('', 'ian'),
    'yang': ('', 'iang'),
    'yao': ('', 'iao'),
    'ye': ('', 'ie'),
    'yi': ('', 'i'),
    'yin': ('', 'in'),
    'ying': ('', 'ing'),
    'yo': ('', 'iou'),
    'yong': ('', 'iong'),
    'you': ('', 'iou'),
    'yu': ('', 'v'),
    'yuan': ('', 'van'),
    'yue': ('', 've'),
    'yun': ('', 'vn'),
    'za': ('z', 'a'),
    'zai': ('z', 'ai'),
    'zan': ('z', 'an'),
    'zang': ('z', 'ang'),
    'zao': ('z', 'ao'),
    'ze': ('z', 'e'),
    'zei': ('z', 'ei'),
    'zen': ('z', 'en'),
    'zeng': ('z', 'eng'),
    'zha': ('zh', 'a'),
    'zhai': ('zh', 'ai'),
    'zhan': ('zh', 'an'),
    'zhang': ('zh', 'ang'),
    'zhao': ('zh', 'ao'),
    'zhe': ('zh', 'e'),
    'zhei': ('zh', 'ei'),
    'zhen': ('zh', 'en'),
    'zheng': ('zh', 'eng'),
    'zhi': ('zh', 'iii'),
    'zhong': ('zh', 'ong'),
    'zhou': ('zh', 'ou'),
    'zhu': ('zh', 'u'),
    'zhua': ('zh', 'ua'),
    'zhuai': ('zh', 'uai'),
    'zhuan': ('zh', 'uan'),
    'zhuang': ('zh', 'uang'),
    'zhui': ('zh', 'uei'),
    'zhun': ('zh', 'uen'),
    'zhuo': ('zh', 'uo'),
    'zi': ('z', 'ii'),
    'zong': ('z', 'ong'),
    'zou': ('z', 'ou'),
    'zu': ('z', 'u'),
    'zuan': ('z', 'uan'),
    'zui': ('z', 'uei'),
    'zun': ('z', 'uen'),
    'zuo': ('z', 'uo'),
}


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


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


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

        self.pinyin = self.get_pinyin()

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
        for x in text.split():
            assert x[-1].isdigit()
            tone = x[-1]
            initial, final = pinyin_dict[x[:-1]]
            if initial != '':
                result.append(initial)
            assert final is not ''
            result.append(final + tone)
        result = ' '.join(result)
        return result

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

    def get_pinyin(self):
        my_pinyin = Pinyin(MyConverter())
        pinyin = my_pinyin.pinyin
        return pinyin

    def text_to_sequence(self, text, inference=False):
        global _symbol_to_id

        if inference:
            text = self.pinyin(text, style=Style.TONE3)
            new_text = []
            for x in text:
                new_text.append(''.join(x))
            text = self.get_initials_and_finals(' '.join(new_text))
            print(text)

        sequence = []
        for symbol in text.split():
            idx = _symbol_to_id[symbol]
            sequence.append(idx)
        return sequence
