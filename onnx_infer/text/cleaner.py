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
    文本序列化
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


def clean_text(text, language):
    norm_text = getattr(text_normalize_instance, language)(text)
    phones, tones, word2ph = getattr(g2p_instance, language)(norm_text)
    return norm_text, phones, tones, word2ph


# def clean_text_bert(text, language):
#     language_module = language_module_map[language]
#     norm_text = language_module.text_normalize(text)
#     phones, tones, word2ph = language_module.g2p(norm_text)
#     bert = language_module.get_bert_feature(norm_text, word2ph)
#     return phones, tones, bert


# def text_to_sequence(text, language):
#     norm_text, phones, tones, word2ph = clean_text(text, language)
#     return cleaned_text_to_sequence(phones, tones, language)


# if __name__ == "__main__":
#     pass
