# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team, Jaehyoung Kim(@crux153) and Taehoon Kim(@carpedm20)
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

# Code based on https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
"""Korean related helpers."""

import ast
import json
import os
import re

from jamo import h2j, hangul_to_jamo, j2h, jamo_to_hcj

etc_dictionary = {
    "2 30대": "이삼십대",
    "20~30대": "이삼십대",
    "20, 30대": "이십대 삼십대",
    "1+1": "원플러스원",
    "3에서 6개월인": "3개월에서 육개월인",
}

english_dictionary = {
    "Devsisters": "데브시스터즈",
    "track": "트랙",
    # krbook
    "LA": "엘에이",
    "LG": "엘지",
    "KOREA": "코리아",
    "JSA": "제이에스에이",
    "PGA": "피지에이",
    "GA": "지에이",
    "idol": "아이돌",
    "KTX": "케이티엑스",
    "AC": "에이씨",
    "DVD": "디비디",
    "US": "유에스",
    "CNN": "씨엔엔",
    "LPGA": "엘피지에이",
    "P": "피",
    "L": "엘",
    "T": "티",
    "B": "비",
    "C": "씨",
    "BIFF": "비아이에프에프",
    "GV": "지비",
    # JTBC
    "IT": "아이티",
    "IQ": "아이큐",
    "JTBC": "제이티비씨",
    "trickle down effect": "트리클 다운 이펙트",
    "trickle up effect": "트리클 업 이펙트",
    "down": "다운",
    "up": "업",
    "FCK": "에프씨케이",
    "AP": "에이피",
    "WHERETHEWILDTHINGSARE": "",
    "Rashomon Effect": "",
    "O": "오",
    "OO": "오오",
    "B": "비",
    "GDP": "지디피",
    "CIPA": "씨아이피에이",
    "YS": "와이에스",
    "Y": "와이",
    "S": "에스",
    "JTBC": "제이티비씨",
    "PC": "피씨",
    "bill": "빌",
    "Halmuny": "하모니",  #####
    "X": "엑스",
    "SNS": "에스엔에스",
    "ability": "어빌리티",
    "shy": "",
    "CCTV": "씨씨티비",
    "IT": "아이티",
    "the tenth man": "더 텐쓰 맨",  ####
    "L": "엘",
    "PC": "피씨",
    "YSDJJPMB": "",  ########
    "Content Attitude Timing": "컨텐트 애티튜드 타이밍",
    "CAT": "캣",
    "IS": "아이에스",
    "K": "케이",
    "Y": "와이",
    "KDI": "케이디아이",
    "DOC": "디오씨",
    "CIA": "씨아이에이",
    "PBS": "피비에스",
    "D": "디",
    "PPropertyPositionPowerPrisonP" "S": "에스",
    "francisco": "프란시스코",
    "I": "아이",
    "III": "아이아이",  ######
    "No joke": "노 조크",
    "BBK": "비비케이",
    "LA": "엘에이",
    "Don": "",
    "t worry be happy": " 워리 비 해피",
    "NO": "엔오",  #####
    "it was our sky": "잇 워즈 아워 스카이",
    "it is our sky": "잇 이즈 아워 스카이",  ####
    "NEIS": "엔이아이에스",  #####
    "IMF": "아이엠에프",
    "apology": "어폴로지",
    "humble": "험블",
    "M": "엠",
    "Nowhere Man": "노웨어 맨",
    "The Tenth Man": "더 텐쓰 맨",
    "PBS": "피비에스",
    "BBC": "비비씨",
    "MRJ": "엠알제이",
    "CCTV": "씨씨티비",
    "Pick me up": "픽 미 업",
    "DNA": "디엔에이",
    "UN": "유엔",
    "STOP": "스탑",  #####
    "PRESS": "프레스",  #####
    "not to be": "낫 투비",
    "Denial": "디나이얼",
    "G": "지",
    "IMF": "아이엠에프",
    "GDP": "지디피",
    "JTBC": "제이티비씨",
    "Time flies like an arrow": "타임 플라이즈 라이크 언 애로우",
    "DDT": "디디티",
    "AI": "에이아이",
    "Z": "제트",
    "OECD": "오이씨디",
    "N": "앤",
    "A": "에이",
    "MB": "엠비",
    "EH": "이에이치",
    "IS": "아이에스",
    "TV": "티비",
    "MIT": "엠아이티",
    "KBO": "케이비오",
    "I love America": "아이 러브 아메리카",
    "SF": "에스에프",
    "Q": "큐",
    "KFX": "케이에프엑스",
    "PM": "피엠",
    "Prime Minister": "프라임 미니스터",
    "Swordline": "스워드라인",
    "TBS": "티비에스",
    "DDT": "디디티",
    "CS": "씨에스",
    "Reflecting Absence": "리플렉팅 앱센스",
    "PBS": "피비에스",
    "Drum being beaten by everyone": "드럼 빙 비튼 바이 에브리원",
    "negative pressure": "네거티브 프레셔",
    "F": "에프",
    "KIA": "기아",
    "FTA": "에프티에이",
    "Que sais-je": "",
    "UFC": "유에프씨",
    "P": "피",
    "DJ": "디제이",
    "Chaebol": "채벌",
    "BBC": "비비씨",
    "OECD": "오이씨디",
    "BC": "삐씨",
    "C": "씨",
    "B": "씨",
    "KY": "케이와이",
    "K": "케이",
    "CEO": "씨이오",
    "YH": "와이에치",
    "IS": "아이에스",
    "who are you": "후 얼 유",
    "Y": "와이",
    "The Devils Advocate": "더 데빌즈 어드보카트",
    "YS": "와이에스",
    "so sorry": "쏘 쏘리",
    "Santa": "산타",
    "Big Endian": "빅 엔디안",
    "Small Endian": "스몰 엔디안",
    "Oh Captain My Captain": "오 캡틴 마이 캡틴",
    "AIB": "에이아이비",
    "K": "케이",
    "PBS": "피비에스",
    # IU
    "ASMR": "에이에스엠알",
    "V": "브이",
    "PD": "피디",
    "CD": "씨디",
    "ANR": "에이엔알",
    "Twenty Three": "투엔티 쓰리",
    "Through The Night": "쓰루 더 나잇",
    "MD": "엠디",
}

num_to_kor = {
    "0": "영",
    "1": "일",
    "2": "이",
    "3": "삼",
    "4": "사",
    "5": "오",
    "6": "육",
    "7": "칠",
    "8": "팔",
    "9": "구",
}

unit_to_kor1 = {"%": "퍼센트", "cm": "센치미터", "mm": "밀리미터", "km": "킬로미터", "kg": "킬로그람"}
unit_to_kor2 = {"m": "미터"}

upper_to_kor = {
    "A": "에이",
    "B": "비",
    "C": "씨",
    "D": "디",
    "E": "이",
    "F": "에프",
    "G": "지",
    "H": "에이치",
    "I": "아이",
    "J": "제이",
    "K": "케이",
    "L": "엘",
    "M": "엠",
    "N": "엔",
    "O": "오",
    "P": "피",
    "Q": "큐",
    "R": "알",
    "S": "에스",
    "T": "티",
    "U": "유",
    "V": "브이",
    "W": "더블유",
    "X": "엑스",
    "Y": "와이",
    "Z": "지",
}


"""
초성과 종성은 같아보이지만, 다른 character이다.

'_-!'(),-.:;? ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ~'

'_': 0, '-': 7, '!': 2, "'": 3, '(': 4, ')': 5, ',': 6, '.': 8, ':': 9, ';': 10,
'?': 11, ' ': 12, 'ᄀ': 13, 'ᄁ': 14, 'ᄂ': 15, 'ᄃ': 16, 'ᄄ': 17, 'ᄅ': 18, 'ᄆ': 19, 'ᄇ': 20,
'ᄈ': 21, 'ᄉ': 22, 'ᄊ': 23, 'ᄋ': 24, 'ᄌ': 25, 'ᄍ': 26, 'ᄎ': 27, 'ᄏ': 28, 'ᄐ': 29, 'ᄑ': 30,
'ᄒ': 31, 'ᅡ': 32, 'ᅢ': 33, 'ᅣ': 34, 'ᅤ': 35, 'ᅥ': 36, 'ᅦ': 37, 'ᅧ': 38, 'ᅨ': 39, 'ᅩ': 40,
'ᅪ': 41, 'ᅫ': 42, 'ᅬ': 43, 'ᅭ': 44, 'ᅮ': 45, 'ᅯ': 46, 'ᅰ': 47, 'ᅱ': 48, 'ᅲ': 49, 'ᅳ': 50,
'ᅴ': 51, 'ᅵ': 52, 'ᆨ': 53, 'ᆩ': 54, 'ᆪ': 55, 'ᆫ': 56, 'ᆬ': 57, 'ᆭ': 58, 'ᆮ': 59, 'ᆯ': 60,
'ᆰ': 61, 'ᆱ': 62, 'ᆲ': 63, 'ᆳ': 64, 'ᆴ': 65, 'ᆵ': 66, 'ᆶ': 67, 'ᆷ': 68, 'ᆸ': 69, 'ᆹ': 70,
'ᆺ': 71, 'ᆻ': 72, 'ᆼ': 73, 'ᆽ': 74, 'ᆾ': 75, 'ᆿ': 76, 'ᇀ': 77, 'ᇁ': 78, 'ᇂ': 79, '~': 80
"""

_pad = "pad"
_eos = "eos"
_punctuation = "!'(),-.:;? "
_special = "-"

_jamo_leads = [chr(_) for _ in range(0x1100, 0x1113)]
_jamo_vowels = [chr(_) for _ in range(0x1161, 0x1176)]
_jamo_tails = [chr(_) for _ in range(0x11A8, 0x11C3)]

_letters = _jamo_leads + _jamo_vowels + _jamo_tails

symbols = [_pad] + list(_special) + list(_punctuation) + _letters + [_eos]

_symbol_to_id = {c: i for i, c in enumerate(symbols)}
_id_to_symbol = {i: c for i, c in enumerate(symbols)}

quote_checker = """([`"'＂“‘])(.+?)([`"'＂”’])"""


def is_lead(char):
    return char in _jamo_leads


def is_vowel(char):
    return char in _jamo_vowels


def is_tail(char):
    return char in _jamo_tails


def get_mode(char):
    if is_lead(char):
        return 0
    elif is_vowel(char):
        return 1
    elif is_tail(char):
        return 2
    else:
        return -1


def _get_text_from_candidates(candidates):
    if len(candidates) == 0:
        return ""
    elif len(candidates) == 1:
        return jamo_to_hcj(candidates[0])
    else:
        return j2h(**dict(zip(["lead", "vowel", "tail"], candidates)))


def jamo_to_korean(text):
    text = h2j(text)

    idx = 0
    new_text = ""
    candidates = []

    while True:
        if idx >= len(text):
            new_text += _get_text_from_candidates(candidates)
            break

        char = text[idx]
        mode = get_mode(char)

        if mode == 0:
            new_text += _get_text_from_candidates(candidates)
            candidates = [char]
        elif mode == -1:
            new_text += _get_text_from_candidates(candidates)
            new_text += char
            candidates = []
        else:
            candidates.append(char)

        idx += 1
    return new_text


def compare_sentence_with_jamo(text1, text2):
    return h2j(text1) != h2j(text2)


def tokenize(text, as_id=False):
    # jamo package에 있는 hangul_to_jamo를 이용하여 한글 string을 초성/중성/종성으로 나눈다.
    text = normalize(text)
    tokens = list(
        hangul_to_jamo(text)
    )  # '존경하는'  --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']

    if as_id:
        return [_symbol_to_id[token] for token in tokens]
    else:
        return [token for token in tokens]


def tokenizer_fn(iterator):
    return (token for x in iterator for token in tokenize(x, as_id=False))


def normalize(text):
    text = text.strip()

    text = re.sub("\(\d+일\)", "", text)
    text = re.sub("\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)", "", text)

    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = re.sub("[a-zA-Z]+", normalize_upper, text)

    text = normalize_quote(text)
    text = normalize_number(text)

    return text


def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        else:
            return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


def normalize_upper(text):
    text = text.group(0)

    if all([char.isupper() for char in text]):
        return "".join(upper_to_kor[char] for char in text)
    else:
        return text


def normalize_quote(text):
    def fn(found_text):
        from nltk import sent_tokenize  # NLTK doesn't along with multiprocessing

        found_text = found_text.group()
        unquoted_text = found_text[1:-1]

        sentences = sent_tokenize(unquoted_text)
        return " ".join(["'{}'".format(sent) for sent in sentences])

    return re.sub(quote_checker, fn, text)


number_checker = "([+-]?\d[\d,]*)[\.]?\d*"
count_checker = "(시|명|가지|살|마리|포기|송이|수|톨|통|점|개|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)"


def normalize_number(text):
    text = normalize_with_dictionary(text, unit_to_kor1)
    text = normalize_with_dictionary(text, unit_to_kor2)
    text = re.sub(
        number_checker + count_checker, lambda x: number_to_korean(x, True), text
    )
    text = re.sub(number_checker, lambda x: number_to_korean(x, False), text)
    return text


num_to_kor1 = [""] + list("일이삼사오육칠팔구")
num_to_kor2 = [""] + list("만억조경해")
num_to_kor3 = [""] + list("십백천")

# count_to_kor1 = [""] + ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
count_to_kor1 = [""] + ["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]

count_tenth_dict = {
    "십": "열",
    "두십": "스물",
    "세십": "서른",
    "네십": "마흔",
    "다섯십": "쉰",
    "여섯십": "예순",
    "일곱십": "일흔",
    "여덟십": "여든",
    "아홉십": "아흔",
}


def number_to_korean(num_str, is_count=False):
    if is_count:
        num_str, unit_str = num_str.group(1), num_str.group(2)
    else:
        num_str, unit_str = num_str.group(), ""

    num_str = num_str.replace(",", "")
    num = ast.literal_eval(num_str)

    if num == 0:
        return "영"

    check_float = num_str.split(".")
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) >= 3:
        raise Exception(" [!] Wrong number format")
    else:
        digit_str, float_str = check_float[0], None

    if is_count and float_str is not None:
        raise Exception(" [!] `is_count` and float number does not fit each other")

    digit = int(digit_str)

    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    kor = ""
    size = len(str(digit))
    tmp = []

    for i, v in enumerate(digit_str, start=1):
        v = int(v)

        if v != 0:
            if is_count:
                tmp += count_to_kor1[v]
            else:
                tmp += num_to_kor1[v]

            tmp += num_to_kor3[(size - i) % 4]

        if (size - i) % 4 == 0 and len(tmp) != 0:
            kor += "".join(tmp)
            tmp = []
            kor += num_to_kor2[int((size - i) / 4)]

    if is_count:
        if kor.startswith("한") and len(kor) > 1:
            kor = kor[1:]

        if any(word in kor for word in count_tenth_dict):
            kor = re.sub(
                "|".join(count_tenth_dict.keys()),
                lambda x: count_tenth_dict[x.group()],
                kor,
            )

    if not is_count and kor.startswith("일") and len(kor) > 1:
        kor = kor[1:]

    if float_str is not None:
        kor += "쩜 "
        kor += re.sub("\d", lambda x: num_to_kor[x.group()], float_str)

    if num_str.startswith("+"):
        kor = "플러스 " + kor
    elif num_str.startswith("-"):
        kor = "마이너스 " + kor

    return kor + unit_str
