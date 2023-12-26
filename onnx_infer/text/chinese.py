import os
import re

import cn2an
import jieba.posseg as psg
from typing import List, Dict
from pypinyin import lazy_pinyin, Style

from .symbols import punctuation
from .chinese_tone_sandhi import ToneSandhi

from log import log_instance

REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


class ChineseG2P:
    def __init__(self) -> None:
        self.tone_modifier = ToneSandhi()
        self.pinyin_to_symbol_map: Dict[str, str] = {}
        self.__read_opencpop_symbol_map()

    def __read_opencpop_symbol_map(self):
        """
        取读opencpop数据
        """
        f = open("onnx/Text/opencpop-strict.txt", "r")
        for line in f.readlines():
            self.pinyin_to_symbol_map[line.split("\t")[0]] = line.strip().split("\t")[1]
        f.close()

    @staticmethod
    def __get_initials_finals(word):
        initials = []
        finals = []
        orig_initials = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.INITIALS
        )
        orig_finals = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
        )
        for c, v in zip(orig_initials, orig_finals):
            initials.append(c)
            finals.append(v)
        return initials, finals

    def g2p(self, segments_list: List[str]):
        phones_list = []
        tones_list = []
        word2ph = []
        for seg in segments_list:
            seg_cut = psg.lcut(seg)
            initials = []
            finals = []
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = self.__get_initials_finals(word)
                sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                initials.append(sub_initials)
                finals.append(sub_finals)

                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            #
            for c, v in zip(initials, finals):
                raw_pinyin = c + v
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c == v:
                    assert c in punctuation
                    phone = [c]
                    tone = "0"
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]

                    pinyin = c + v_without_tone
                    assert tone in "12345"

                    if c:
                        # 多音节
                        v_rep_map = {
                            "uei": "ui",
                            "iou": "iu",
                            "uen": "un",
                        }
                        if v_without_tone in v_rep_map.keys():
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        # 单音节
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                    assert pinyin in self.pinyin_to_symbol_map.keys(), (
                        pinyin,
                        seg,
                        raw_pinyin,
                    )
                    phone = self.pinyin_to_symbol_map[pinyin].split(" ")
                    word2ph.append(len(phone))

                phones_list += phone
                tones_list += [int(tone)] * len(phone)
        return phones_list, tones_list, word2ph


chinese_g2p_instance = ChineseG2P()


def g2p(text: str):
    """
    将文本转换成音节
    """
    # 将文本按照标点符号切分成列表
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # 根据切分后的列表，返回文本对应发音列表
    # phone:拼音的声母、韵母
    # tone:声调 1 2 3 4 5
    # word2ph:如果只有韵母，返回1，如果有声母韵母，返回2
    phones_list, tones_list, word2ph_list = chinese_g2p_instance.g2p(sentences)
    if sum(word2ph_list) != len(phones_list):
        raise ValueError("中文转拼音失败：音节总数(sum(word2ph_list))与音节的个数(len(phones_list))不匹配。")
    if len(word2ph_list) != len(text):  # Sometimes it will crash,you can add a try-catch.
        raise ValueError("中文转拼音失败：拼音结果个数(len(word2ph_list))与文本长度(len(text))不匹配。")

    phones_list = ["_"] + phones_list + ["_"]
    log_instance.debug(f"phones {str(phones_list)}")
    tones_list = [0] + tones_list + [0]
    log_instance.debug(f"tones {str(tones_list)}")
    word2ph_list = [1] + word2ph_list + [1]
    log_instance.debug(f"word2ph {str(word2ph_list)}")
    return phones_list, tones_list, word2ph_list


def replace_punctuation(text: str):
    """
    替换所有中文标点符号为指定的英文符号： ["!", "?", "…", ",", ".", "'", "-"]
    """
    # 替换某些同音字
    text = text.replace("嗯", "恩").replace("呣", "母")
    # 将所有标点符号替换成指定英文符号
    pattern = re.compile("|".join(re.escape(p) for p in REP_MAP.keys()))
    replaced_text = pattern.sub(lambda x: REP_MAP[x.group()], text)
    # 剔除非指定英文符号和中文的所有字符
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )
    return replaced_text


def text_normalize(text: str):
    """
    替换所有阿拉伯数字为中文，同时将中文符号替换为英文符号
    """
    # 提取文本中所有的阿拉伯数字
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        # 将阿拉伯数字转中文小写数字一百二十三
        text = text.replace(number, cn2an.an2cn(number), 1)
    # 替换所有中文标点符号为指定的英文符号： ["!", "?", "…", ",", ".", "'", "-"]
    text = replace_punctuation(text)
    return text
