from .symbols import symbol_to_id, language_tone_start_map, language_id_map

from typing import Callable
from dataclasses import dataclass

from .chinese import text_normalize as zh_text_normalize
from .japanese import text_normalize as jp_text_normalize
from .english import text_normalize as en_text_normalize

from .chinese import g2p as zh_g2p
from .japanese import g2p as jp_g2p
from .english import g2p as en_g2p


# from text import cleaned_text_to_sequence
@dataclass
class TextNormalizeDict:
    """
    文本序列化 替换所有阿拉伯数字为对应语言，同时将符号替换为指定列表内的英文符号
    """

    ZH: Callable = zh_text_normalize
    JP: Callable = jp_text_normalize
    EN: Callable = en_text_normalize


@dataclass
class G2PDict:
    """
    文本序列化
    """

    ZH: Callable = zh_g2p
    JP: Callable = jp_g2p
    EN: Callable = en_g2p


text_normalize_instance = TextNormalizeDict()
g2p_instance = G2PDict()


def clean_text(text: str, language: str):
    """
    处理标点符号，并将文本转化成对应语言音标？

    norm_text：处理标点后的文本

    phones：所有文本的音标列表

    tones：所有文本的音调

    word2ph：单个字的音标个数

    """
    try:
        language_text_normalize = getattr(text_normalize_instance, language)
    except AttributeError:
        raise TypeError(f"语言类型输入错误：{language}。")
    # 替换所有阿拉伯数字为对应语言，同时将符号替换为指定列表内的英文符号
    norm_text = language_text_normalize(text)
    phones, tones, word2ph = getattr(g2p_instance, language)(norm_text)
    return norm_text, phones, tones, word2ph


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_ids = [language_id_map[language]] * len(phones)
    return phones, tones, lang_ids