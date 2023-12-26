import os
from log import log_instance
import numpy as np
from typing import Callable
import onnxruntime as ort
from dataclasses import dataclass

from config import config_instance

from .text.japanese import text2sep_kata as japanese_text2sep_kata

from .text.chinese import text_normalize as chinese_text_normalize
from .text.japanese import text_normalize as japanese_text_normalize
from .text.english import text_normalize as english_text_normalize

from .text.symbols import _symbol_to_id, language_tone_start_map, language_id_map
from .text.tokenizer import tokenizer_instance

ONNX_PROVIDERS = [config_instance.get("onnx_providers", "CPUExecutionProvider")]
CHINESE_ONNX_LOCAL_DIR = config_instance.get("bert_chinese", "")
JAPANESE_ONNX_LOCAL_DIR = config_instance.get("bert_japanese", "")
ENGLISH_ONNX_LOCAL_DIR = config_instance.get("bert_english", "")


@dataclass
class LanguageModulesDict:
    """
    语言方法字典
    """

    ZH: Callable = chinese_text_normalize
    JP: Callable = japanese_text_normalize
    EN: Callable = english_text_normalize


@dataclass
class BertModelsDict:
    """
    ONNX模型对象字典
    """

    ZH: ort.InferenceSession = (
        ort.InferenceSession(
            os.path.join(CHINESE_ONNX_LOCAL_DIR, "model.onnx"),
            providers=ONNX_PROVIDERS,
        )
        if CHINESE_ONNX_LOCAL_DIR
        else None
    )
    JP: ort.InferenceSession = (
        ort.InferenceSession(
            os.path.join(JAPANESE_ONNX_LOCAL_DIR, "model.onnx"),
            providers=ONNX_PROVIDERS,
        )
        if JAPANESE_ONNX_LOCAL_DIR
        else None
    )
    EN: ort.InferenceSession = (
        ort.InferenceSession(
            os.path.join(ENGLISH_ONNX_LOCAL_DIR, "model.onnx"),
            providers=ONNX_PROVIDERS,
        )
        if ENGLISH_ONNX_LOCAL_DIR
        else None
    )


class BertOnnx:
    def __init__(self) -> None:
        log_instance.info("正在加载BERT语言模型...")
        self.models_dict: BertModelsDict = BertModelsDict()

    def __get_model_path(self, dir_path: str):
        """
        获取模型的完整路径
        """
        return os.path.join(dir_path, "model.onnx")

    def __check_language(self, language_str: str):
        """
        检查语言类型参数是否合法
        """
        if hasattr(self.models_dict, language_str):
            return language_str
        return False

    @staticmethod
    def __check_params_inputs(text: str, word2ph: list, language_str: str = "ZH"):
        # 检查输入参数的合法性
        log_instance.debug(f"{language_str}, {str(len(word2ph))} {str(len(text) + 2)}")
        if language_str in ["ZH"] and len(word2ph) != len(text) + 2:
            raise ValueError("输入参数错误，len(word2ph) != len(text) + 2。")
        else:
            pass

    @staticmethod
    def __check_onnx_outputs(res: np.float32, word2ph: list, language_str: str = "ZH"):
        """
        检查输出结果的合法性
        """
        # 检查输出结果的合法性
        if language_str == "EN" and len(word2ph) != res.shape[0]:
            raise ValueError(
                f"输入参数错误，len(word2ph) != res.shape[0] （len(word2ph):{len(word2ph)} res.shape[0]:{res.shape[0]}）。 "
            )
        else:
            pass

    @staticmethod
    def __handle_text(text: str, language_str: str = "ZH"):
        """
        针对不同的语言，对文本进行处理
        """
        log_instance.debug(f"文本处理 {language_str} {text}")
        if language_str == "JP":
            return "".join(japanese_text2sep_kata(text)[0])
        return text

    def __structure_onnx_inputs(self, text: str, language_str: str = "ZH"):
        """
        构造分析器转换文本参数
        """
        # 加载分析器转换文本参数
        tokenizer = getattr(tokenizer_instance, language_str)

        if not tokenizer:
            raise KeyError(f"BERT_{language_str}分析器尚未载入。")

        tokenized_tokens = tokenizer(text)

        # 构造模型 输入参数
        input_feed = {
            "input_ids": np.array([tokenized_tokens["input_ids"]], dtype=np.int64),
            "attention_mask": np.array(
                [tokenized_tokens["attention_mask"]], dtype=np.int64
            ),
        }
        # 目前只有ZH的bert模型需要 token_type_ids 参数
        if language_str == "ZH":
            input_feed["token_type_ids"] = np.array(
                [tokenized_tokens["token_type_ids"]], dtype=np.int64
            )

        return input_feed

    def __get_phone_level_feature(self, res: np.float32, word2ph: list) -> np.float32:
        """
        获取最终bert结果 ？？？具体作用不清楚
        """
        phone_level_feature = []
        for i in range(len(word2ph)):
            if i >= res.shape[0]:
                repeat_feature = np.repeat([np.empty(res.shape[1])], word2ph[i], axis=0)
            else:
                repeat_feature = np.repeat([res[i]], word2ph[i], axis=0)
            phone_level_feature.append(repeat_feature)
        res = np.concatenate(phone_level_feature, axis=0)
        # 强制转化为float32
        res = res.astype(np.float32)
        return res

    def __run_onnx(self, inputs: dict, language_str: str = "ZH") -> np.float32:
        """
        输入并获取bert最后一层隐藏数据
        """
        onnx_model: ort.InferenceSession = getattr(self.models_dict, language_str)

        if not onnx_model:
            raise KeyError(f"BERT_{language_str}模型尚未载入。")
        res = onnx_model.run(
            output_names=["last_hidden_state"],
            input_feed=inputs,
        )[0]
        onnx_model.disable_fallback()
        res = np.array(res, dtype=np.float32)
        return res

    def run(self, norm_text: str, word2ph: list, language_str: str) -> np.float32:
        """
        运行推理
        """
        # 检查语言类型参数是否合法
        if not self.__check_language(language_str):
            raise TypeError(f"语言类型输入错误：{language_str}。")

        # 针对不同的语言，对文本进行处理
        norm_text = self.__handle_text(text=norm_text, language_str=language_str)
        log_instance.debug(f"结果处理 {language_str} {norm_text}")
        # 检查输入参数
        self.__check_params_inputs(
            text=norm_text, word2ph=word2ph, language_str=language_str
        )

        # 构造模型输入参数
        input_feed = self.__structure_onnx_inputs(
            text=norm_text, language_str=language_str
        )

        # 推理获取输出
        res = self.__run_onnx(inputs=input_feed, language_str=language_str)
        log_instance.debug(f"原始onnx输出 {language_str} {str(res.dtype)}")
        # 检查输出结果的合法性
        self.__check_onnx_outputs(res=res, word2ph=word2ph, language_str=language_str)

        # 获取最终bert结果 ？？？具体作用不清楚
        phone_level_feature = self.__get_phone_level_feature(res=res, word2ph=word2ph)
        log_instance.debug(f"最终bert结果 {language_str} {str(phone_level_feature.dtype)}")
        return phone_level_feature


bert_onnx_instance = BertOnnx()
language_modules_instance = LanguageModulesDict()


def clean_text(text, language: str):
    try:
        language_module = getattr(language_modules_instance, language)
    except AttributeError:
        raise TypeError(f"语言类型输入错误：{language}。")
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text: str, word2ph: list, language: np.int64):
    """
    bert模型推理
    """
    return bert_onnx_instance.run(norm_text, word2ph, language)
