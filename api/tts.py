import os
import numpy as np
from uuid import uuid4
from log import log_instance
from config import config_instance
from scipy.io import wavfile
from typing import Callable, List
from dataclasses import dataclass
from onnx_infer.onnx_infer import infor_onnx_instance

from .split import split_text, text_split_to_sentence
from .utils import rebuild_temp_dir

EMOTION = 7
SDP_RATIO = 0.2
NOISE = 0.6
NOISEW = 0.8
LENGTH = 0.8
LANGUAGE = "ZH"
AUDIO_RATE = 44100

TEMP_PATH = os.path.abspath("./temp")


def change_to_wav(
    file_path: str, data: np.float32, sample_rate: int = AUDIO_RATE
) -> str:
    """
    将返回的numpy数据转换成音频
    """
    scaled_data = np.int16(data * 32767)
    wavfile.write(file_path, sample_rate, scaled_data)
    return file_path


def __generate_empty_float32(sample_rate: int = AUDIO_RATE) -> tuple:
    """
    生成空音频的numpy数据
    """
    return tuple(
        sample_rate,
        np.concatenate([np.zeros(sample_rate // 2)]),
    )


def __generate_slient_audio(
    interval_time: float = 1.5, sample_rate: int = AUDIO_RATE
) -> np.float32:
    """
    生成指定秒数的空音频数据
    """
    return np.zeros((int)(sample_rate * interval_time), dtype=np.float32).reshape(
        1, 1, int(sample_rate * interval_time)
    )


def __generate_single_audio(
    text: str,
    speaker_name: str,
    language: str = "ZH",
    sdp_ratio: float = SDP_RATIO,
    noise_scale: float = NOISE,
    noise_scale_w: float = NOISEW,
    length_scale: float = LENGTH,
    emotion: float = EMOTION,
    seed: int = 114514,
) -> np.float32:
    """
    根据text生成单语言音频
    """
    audio = infor_onnx_instance.infer(
        text=text,
        speaker_name=speaker_name,
        language=language,
        sdp_ratio=sdp_ratio,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        length_scale=length_scale,
        emotion=emotion,
        seed=seed,
    )
    return audio


def __generate_multilang_audio(
    text: str,
    speaker_name: str,
    sdp_ratio: float = SDP_RATIO,
    noise_scale: float = NOISE,
    noise_scale_w: float = NOISEW,
    length_scale: float = LENGTH,
    emotion: float = EMOTION,
    seed: int = 114514,
) -> np.float32:
    """
    根据text自动切分，生成多语言混合音频
    """
    text_list, language_list = split_text(text)

    if not language_list:
        log_instance.warning("文本转语音推理失败：{speaker_name} -> {text} 文本内容不可为空。")
        return __generate_empty_float32()

    elif len(language_list) == 1:
        audio = infor_onnx_instance.infer(
            text=text_list[0],
            speaker_name=speaker_name,
            language=language_list[0],
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            emotion=emotion,
            seed=seed,
        )
    else:
        audio = infor_onnx_instance.infer_multilang(
            text_list=text_list,
            speaker_name=speaker_name,
            language_list=language_list,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            emotion=emotion,
            seed=seed,
        )
    return audio


def __generate_multi_within(
    text_list: List[str],
    speaker_name: str,
    language: str = "ZH",
    sdp_ratio: float = SDP_RATIO,
    noise_scale: float = NOISE,
    noise_scale_w: float = NOISEW,
    length_scale: float = LENGTH,
    emotion: float = EMOTION,
    seed: int = 114514,
    within_interval: float = 0.5,
) -> np.float32:
    """
    根据多个句内文字段生成语音
    """
    # 获取局部变量
    params_dict: dict = locals()
    del params_dict["text_list"]
    del params_dict["within_interval"]

    within_audio_list = []
    list_length = len(text_list)

    for index, text in enumerate(text_list):
        params_dict["text"] = text
        log_instance.info(
            f"正在推理({str(index+1)}/{str(list_length)})：{speaker_name} -> {text}"
        )

        # 判断是否需要自动多语言切分
        if language.lower() == "auto":
            try:
                del params_dict["language"]
            except KeyError:
                pass
            audio = __generate_multilang_audio(**params_dict)
        else:
            params_dict["language"] = language
            audio = __generate_single_audio(**params_dict)

        # 将所有语音句子数据存入列表中
        within_audio_list.append(audio)
        # 插入静音数据
        slient_audio = __generate_slient_audio(interval_time=within_interval)
        within_audio_list.append(slient_audio)

    # 删除最后一个静音数据
    within_audio_list.pop()
    # 将列表中的语音数据合成
    audio_concat = np.concatenate(within_audio_list, axis=2)

    return audio_concat


def __generate_multi_sentence(
    text_list: List[str],
    speaker_name: str,
    language: str = "ZH",
    sdp_ratio: float = SDP_RATIO,
    noise_scale: float = NOISE,
    noise_scale_w: float = NOISEW,
    length_scale: float = LENGTH,
    emotion: float = EMOTION,
    seed: int = 114514,
    within_interval: float = 0.5,
    sentence_interval: float = 1.0,
) -> np.float32:
    """
    根据多个句子生成语音
    """
    # 获取局部变量
    params_dict: dict = locals()
    del params_dict["text_list"]
    del params_dict["sentence_interval"]

    sentence_audio_list = []
    for whithin_text_list in text_list:
        # 句子列表数据合成一个段落音频数据
        params_dict["text_list"] = whithin_text_list
        sentence_audio = __generate_multi_within(**params_dict)
        sentence_audio_list.append(sentence_audio)
        # 插入静音数据
        slient_audio = __generate_slient_audio(interval_time=sentence_interval)
        sentence_audio_list.append(slient_audio)
    # 删除最后一个静音数据
    sentence_audio_list.pop()
    audio_concat = np.concatenate(sentence_audio_list, axis=2)
    return audio_concat


def generate_tts_auto(
    text: str,
    speaker_name: str,
    language: str = "ZH",
    sdp_ratio: float = 0.2,
    noise_scale: float = 0.6,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    emotion: int = 7,
    seed: int = 114514,
    within_interval: float = 0.5,
    sentence_interval: float = 1.0,
    paragraph_interval: float = 2.0,
) -> np.float32:
    """
    自动切分，生成语音
    """
    # 获取局部变量
    params_dict: dict = locals()
    del params_dict["text"]
    del params_dict["paragraph_interval"]

    # 根据文本进行按句子切分成三级列表
    paragraph_sentences_text_list = text_split_to_sentence(text=text)
    log_instance.debug(f"自动切分结果 {str(paragraph_sentences_text_list)}")
    # 检测文本是否为空，为空直接返回空音频
    if len(paragraph_sentences_text_list) == 0:
        log_instance.warning("文本转语音推理失败：{speaker_name} -> {text} 文本内容不可为空。")
        return __generate_empty_float32()

    # 获取每一个段落所有句子的语音数据
    paragraph_audio_list = []
    for sentences_text_list in paragraph_sentences_text_list:
        # 句子列表数据合成一个段落音频数据
        params_dict["text_list"] = sentences_text_list
        paragraph_audio = __generate_multi_sentence(**params_dict)
        paragraph_audio_list.append(paragraph_audio)
        # 插入静音数据
        slient_audio = __generate_slient_audio(interval_time=paragraph_interval)
        paragraph_audio_list.append(slient_audio)
    # 删除最后一个静音数据
    paragraph_audio_list.pop()
    audio_concat = np.concatenate(paragraph_audio_list, axis=2)
    return audio_concat


@dataclass
class InferHander:
    single: Callable = None
    auto: Callable = None


class GenerateTTS:
    def __init__(self) -> None:
        # 重建语音缓存文件夹
        rebuild_temp_dir(TEMP_PATH)
        # 加载onnx推理实例
        self.onnx_infer = infor_onnx_instance

    def get_speakers_list(self, chinese_only: bool = True) -> list:
        """
        获取处理过后的角色列表
        """

        if not chinese_only:
            return self.onnx_infer.speakers_list

        speakers_list = []
        chinese_mark = config_instance.get("onnx_tts_models_chinese_mark", "中文")

        for speaker_name in self.onnx_infer.speakers_list:
            if chinese_mark not in speaker_name:
                continue
            speakers_list.append(
                speaker_name.replace(chinese_mark, "")
                .replace("-", "")
                .replace("_", "")
                .replace("(", "[")
                .replace(")", "]")
            )
        return speakers_list

    def gen_tts(
        self,
        text: str,
        speaker_name: str,
        language: str = "ZH",
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        emotion: int = 7,
        seed: int = 114514,
        within_interval: float = 0.5,
        sentence_interval: float = 1.0,
        paragraph_interval: float = 2.0,
    ):
        """
        tts生成
        """
        # 获取传入参数
        params_dict: dict = locals()
        del params_dict["self"]

        audio = generate_tts_auto(**params_dict)

        file_path = os.path.join(TEMP_PATH, uuid4().hex + ".wav")
        file_path = change_to_wav(file_path, audio)
        return file_path


tts_instance = GenerateTTS()
