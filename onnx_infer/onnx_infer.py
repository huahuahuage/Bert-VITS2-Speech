import os
import numpy as np
from copy import copy
from typing import List
from dataclasses import dataclass
import onnxruntime as ort

from log import log_instance
from config import read_config
from config import config_instance
from .text.cleaner import clean_text, cleaned_text_to_sequence

BERT_ENABLE = config_instance.get("bert_enable", True)

if BERT_ENABLE:
    from .onnx_bert import get_bert


# 获取模型中包含的中文角色标记
CHINESE_CHARACTER_MARK = config_instance.get("onnx_tts_models_chinese_mark", "中文")

ONNX_PROVIDERS = [config_instance.get("onnx_providers", "CPUExecutionProvider")]
MODELS_PATH = os.path.abspath(config_instance.get("onnx_tts_models", "onnx/models"))
MODELS_BASE_NAME = os.path.basename(MODELS_PATH)
MODELS_PARENT_PATH = os.path.dirname(MODELS_PATH)
MODELS_PREFIX = os.path.join(MODELS_PATH, os.path.basename(MODELS_PATH))

ONNX_MODELS_PATH = {
    "config": f"{MODELS_PARENT_PATH}/{MODELS_BASE_NAME}.json",
    "enc": f"{MODELS_PREFIX}_enc_p.onnx",
    "emb_g": f"{MODELS_PREFIX}_emb.onnx",
    "dp": f"{MODELS_PREFIX}_dp.onnx",
    "sdp": f"{MODELS_PREFIX}_sdp.onnx",
    "flow": f"{MODELS_PREFIX}_flow.onnx",
    "dec": f"{MODELS_PREFIX}_dec.onnx",
}


@dataclass
class ONNX_MODELS:
    enc: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["enc"], providers=ONNX_PROVIDERS
    )
    emb_g: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["emb_g"], providers=ONNX_PROVIDERS
    )
    dp: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["dp"], providers=ONNX_PROVIDERS
    )
    sdp: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["sdp"], providers=ONNX_PROVIDERS
    )
    flow: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["flow"], providers=ONNX_PROVIDERS
    )
    dec: ort.InferenceSession = ort.InferenceSession(
        ONNX_MODELS_PATH["dec"], providers=ONNX_PROVIDERS
    )


class ONNX_RUNTINE:
    def __init__(self):
        log_instance.info("正在加载BERT-VITS语音模型...")
        self.config = read_config(ONNX_MODELS_PATH["config"])
        self.models = ONNX_MODELS()

    def __call__(
        self,
        seq: np.int64,
        tone: np.int64,
        language_id: np.int64,
        bert_zh: np.float32,
        bert_jp: np.float32,
        bert_en: np.float32,
        speaker_id: int,
        seed: int = 114514,
        seq_noise_scale: float = 0.8,
        sdp_noise_scale: float = 0.6,
        length_scale: float = 1.0,
        sdp_ratio: float = 0.2,
        emotion: int = 0,
    ):
        speaker_id: np.int64 = np.array([speaker_id], dtype=np.int64)
        emotion: np.int64 = np.array([emotion], dtype=np.int64)

        # emb_g模型推理
        g = self.models.emb_g.run(None, {"sid": speaker_id})[0]
        g = np.expand_dims(g, -1)

        enc_rtn: List[np.float32] = self.models.enc.run(
            output_names=None,
            input_feed={
                "x": seq,
                "t": tone,
                "language": language_id,
                "bert_0": bert_zh,
                "bert_1": bert_jp,
                "bert_2": bert_en,
                "g": g,
                "sid": speaker_id,
                "vqidx": emotion,
            },
        )

        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
        # 设置随机种子
        np.random.seed(seed)
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale

        logw = self.models.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.models.dp.run(
            None, {"x": x, "x_mask": x_mask, "g": g}
        )[
            0
        ] * (
            1 - sdp_ratio
        )

        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)

        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p: np.float32 = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1))
        m_p = m_p.transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        logs_p: np.float32 = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1))
        logs_p = logs_p.transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p: np.float32 = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        log_instance.debug(
            f"flow模型输入 {str(z_p.shape)} {str(y_mask.shape)} {str(g.shape)}"
        )

        if z_p.shape[2] == 0 or y_mask.shape[2] == 0:
            raise MemoryError("flow模型输入参数错误，有可能是临时缓存空间不足。")

        z: np.float32 = self.models.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]

        return self.models.dec.run(None, {"z_in": z, "g": g})[0]

    def get_config(self, key: str, default=None):
        """
        获取模型配置项
        """
        return self.config.get(key, default)


onnx_runtime_instance = ONNX_RUNTINE()


def __add_blank(phone, tone, language, word2ph):
    """
    ？？添加空白间隔
    """
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1

    return (
        intersperse(phone, 0),
        intersperse(tone, 0),
        intersperse(language, 0),
        word2ph,
    )


def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path


def intersperse(lst, item):
    """
    在列表的每个元素之间插入一个分隔符元素

    如： [1, '-', 2, '-', 3, '-', 4]
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(text: str, language: str, add_blank: bool = True) -> tuple:
    """
    推理前文本预处理
    """
    language_list = ["ZH", "JP", "EN"]
    try:
        language: str = language.upper()
        language_index = language_list.index(language)
        language_str = copy(language)
    except ValueError:
        raise TypeError(f"语言类型输入错误：{language}。")

    norm_text, phone, tone, word2ph = clean_text(text, language)
    # print(norm_text, phone, tone, word2ph)
    # 将phone, tone, language转化为对应id表示
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language)

    # ？？添加空白间隔
    if add_blank:
        phone, tone, language, word2ph = __add_blank(phone, tone, language, word2ph)
    # print(len(phone), sum(word2ph))

    bert_list: list = [np.zeros([len(phone), 1024], dtype=np.float32)] * len(
        language_list
    )

    if BERT_ENABLE and language_str == "ZH":
        bert_ori: np.float32 = get_bert(norm_text, word2ph, language=language_str)
        if bert_ori.shape[0] != len(phone):
            raise KeyError("BERT推理结果与预期不符合。")

        bert_list[language_index] = bert_ori

    del word2ph

    res_tuple = tuple(bert_list) + (
        np.expand_dims(np.array(phone, dtype=np.int64), 0),
        np.expand_dims(np.array(tone, dtype=np.int64), 0),
        np.expand_dims(np.array(language, dtype=np.int64), 0),
    )
    return res_tuple


class INFER_ONNX:
    """
    语音推理实现
    """

    def __init__(self) -> None:
        self.onnx_runtime_instance: ONNX_RUNTINE = onnx_runtime_instance
        self.speakers_list = self.onnx_runtime_instance.get_config(
            "Characters", default=[]
        )

    def get_speaker_id(self, speaker_name: str, chinese_only: bool = True) -> int:
        """
        获取发音人名字对应id，默认仅匹配名字带中文标志的模型，中文标志由 CHINESE_CHARACTER_MARK 决定
        """
        for index, speaker in enumerate(self.speakers_list):
            if speaker_name not in speaker:
                continue
            if chinese_only and CHINESE_CHARACTER_MARK not in speaker:
                continue
            else:
                return index
        return -1

    @staticmethod
    def __clamp(
        value: int | float, min_value: int | float = 0, max_value: int | float = 9
    ):
        """
        限定数据在范围内，超出仅取边缘值
        """
        return max(min_value, min(max_value, value))

    @staticmethod
    def __skip_start(phones, tones, language_id, zh_bert, jp_bert, en_bert):
        """
        ？跳过第一个一个元素
        """
        phones = np.delete(phones, 0, axis=1)
        tones = np.delete(tones, 0, axis=1)
        language_id = np.delete(language_id, 0, axis=1)
        zh_bert = np.delete(zh_bert, 0, axis=0)
        jp_bert = np.delete(jp_bert, 0, axis=0)
        en_bert = np.delete(en_bert, 0, axis=0)
        return phones, tones, language_id, zh_bert, jp_bert, en_bert

    @staticmethod
    def __skip_end(phones, tones, language_id, zh_bert, jp_bert, en_bert):
        """
        ？跳过最后一个元素
        """
        phones = phones[:, :-1]
        tones = tones[:, :-1]
        language_id = language_id[:, :-1]
        zh_bert = zh_bert[:-1, :]
        jp_bert = jp_bert[:-1, :]
        en_bert = en_bert[:-1, :]
        return phones, tones, language_id, zh_bert, jp_bert, en_bert

    def __params_specification(
        self,
        sdp_ratio: float,
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float,
        emotion: int,
    ):
        """
        规范化语音调整参数
        """
        sdp_ratio = self.__clamp(sdp_ratio, min_value=0.0, max_value=1.0)
        noise_scale = self.__clamp(noise_scale, min_value=0.0, max_value=2.0)
        noise_scale_w = self.__clamp(noise_scale_w, min_value=0.1, max_value=2.0)
        length_scale = self.__clamp(length_scale, min_value=0.1, max_value=2.0)
        emotion = self.__clamp(emotion)
        return (sdp_ratio, noise_scale, noise_scale_w, length_scale, emotion)

    def __text_to_model_inputs(
        self,
        text: str,
        language: str = "ZH",
        skip_start: bool = False,
        skip_end: bool = False,
        add_blank: bool = True,
    ):
        """
        将文本转化为onnx模型所需numpy数据
        """
        # 在此处实现当前版本的推理
        # 文本预处理
        zh_bert, jp_bert, en_bert, phones, tones, language_id = get_text(
            text=text, language=language, add_blank=add_blank
        )

        if skip_start:
            # ？跳过第一个一个元素
            phones, tones, language_id, zh_bert, jp_bert, en_bert = self.__skip_start(
                phones, tones, language_id, zh_bert, jp_bert, en_bert
            )

        if skip_end:
            # ？跳过最后一个元素
            phones, tones, language_id, zh_bert, jp_bert, en_bert = self.__skip_end(
                phones, tones, language_id, zh_bert, jp_bert, en_bert
            )

        return phones, tones, language_id, zh_bert, jp_bert, en_bert

    def infer(
        self,
        text: str,
        speaker_name: str,
        language: str = "ZH",
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.8,
        noise_scale_w: float = 0.6,
        length_scale: float = 1.0,
        emotion: int = 7,
        seed: int = 114514,
        skip_start: bool = False,
        skip_end: bool = False,
        add_blank: bool = True,
    ) -> np.float32:
        """
        语音推理
        """
        # 参数规范化
        (
            sdp_ratio,
            noise_scale,
            noise_scale_w,
            length_scale,
            emotion,
        ) = self.__params_specification(
            sdp_ratio, noise_scale, noise_scale_w, length_scale, emotion
        )
        # 到 speakers_map.json 内查找是否存在对应关系，如果有，则返回对应发音人真实名称
        full_speaker_name = speaker_name
        log_instance.debug(f"获取发音人真实名称 {speaker_name} -> {full_speaker_name}")
        
        speaker_id = self.get_speaker_id(full_speaker_name,chinese_only = True)

        if speaker_id == -1:
            raise ValueError(f"无法在模型中找到发音人信息：{speaker_name}。")

        # 文本预处理
        # 将文本转化为onnx模型所需numpy数据
        (
            phones,
            tones,
            language_id,
            zh_bert,
            jp_bert,
            en_bert,
        ) = self.__text_to_model_inputs(
            text=text,
            language=language,
            skip_start=skip_start,
            skip_end=skip_end,
            add_blank=add_blank,
        )
        log_instance.debug(f"推理 {full_speaker_name} -> {text}")
        np_audio = self.onnx_runtime_instance(
            seq=phones,
            tone=tones,
            language_id=language_id,
            bert_zh=zh_bert,
            bert_jp=jp_bert,
            bert_en=en_bert,
            speaker_id=speaker_id,
            seed=seed,
            seq_noise_scale=noise_scale,
            sdp_noise_scale=noise_scale_w,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            emotion=emotion,
        )

        del (
            phones,
            tones,
            language_id,
            zh_bert,
            jp_bert,
            en_bert,
            noise_scale,
            noise_scale_w,
            length_scale,
            sdp_ratio,
            emotion,
        )

        return np_audio

    def infer_multilang(
        self,
        text_list: list,
        speaker_name: str,
        language_list: list = ["ZH"],
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.8,
        noise_scale_w: float = 0.6,
        length_scale: float = 1.0,
        emotion: int = 7,
        seed: int = 114514,
        skip_start: bool = False,
        skip_end: bool = False,
        add_blank: bool = True,
    ) -> np.float32:
        """
        语音混合推理
        """
        # (
        #     zh_bert_list,
        #     jp_bert_list,
        #     en_bert_list,
        #     phones_list,
        #     tones_list,
        #     language_id_list,
        # ) = ([], [], [], [], [], [])

        # # 将所有数据合成到列表中
        # for idx, (text, language) in enumerate(zip(text_list, language_list)):
        #     # 计算skip_start、skip_end参数值
        #     skip_start = (idx != 0) or (skip_start and idx == 0)
        #     skip_end = (idx != len(text_list) - 1) or (
        #         skip_end and idx == len(text_list) - 1
        #     )

        #     # 预处理
        #     (
        #         temp_phones,
        #         temp_tones,
        #         temp_language_id,
        #         temp_zh_bert,
        #         temp_jp_bert,
        #         temp_en_bert,
        #     ) = self.__text_to_model_inputs(
        #         text=text, language=language, add_blank=add_blank
        #     )

        #     zh_bert_list.append(temp_zh_bert)
        #     jp_bert_list.append(temp_jp_bert)
        #     en_bert_list.append(temp_en_bert)
        #     phones_list.append(temp_phones)
        #     tones_list.append(temp_tones)
        #     language_id_list.append(temp_language_id)

        # zh_bert = np.concatenate(zh_bert_list, axis=0)
        # jp_bert = np.concatenate(jp_bert_list, axis=0)
        # en_bert = np.concatenate(en_bert_list, axis=0)
        # phones = np.concatenate(phones_list, axis=1)
        # tones = np.concatenate(tones_list, axis=1)
        # language_id = np.concatenate(language_id_list, axis=1)

        # # 参数规范化
        # (
        #     sdp_ratio,
        #     noise_scale,
        #     noise_scale_w,
        #     length_scale,
        #     emotion,
        # ) = self.__params_specification(
        #     sdp_ratio, noise_scale, noise_scale_w, length_scale, emotion
        # )

        # full_speaker_name = self.get_full_speaker_name(
        #     speaker_name=speaker_name, language_str=language, default=speaker_name
        # )
        # log_instance.info(f"获取发音人真实名称 {speaker_name} -> {full_speaker_name}")

        # speaker_id = self.get_speaker_id(
        #     full_speaker_name,
        #     True if language == "ZH" else False,
        # )
        # speaker_id = self.get_speaker_id(speaker_name=speaker_name)

        # np_audio = self.onnx_runtime_instance(
        #     seq=phones,
        #     tone=tones,
        #     language_id=language_id,
        #     bert_zh=zh_bert,
        #     bert_jp=jp_bert,
        #     bert_en=en_bert,
        #     speaker_id=speaker_id,
        #     seed=seed,
        #     seq_noise_scale=noise_scale,
        #     sdp_noise_scale=noise_scale_w,
        #     length_scale=length_scale,
        #     sdp_ratio=sdp_ratio,
        #     emotion=emotion,
        # )

        # del (
        #     phones,
        #     tones,
        #     language_id,
        #     zh_bert,
        #     jp_bert,
        #     en_bert,
        #     noise_scale,
        #     noise_scale_w,
        #     length_scale,
        #     sdp_ratio,
        #     emotion,
        # )

        audil_list = []

        for idx, (text, language) in enumerate(zip(text_list, language_list)):
            # 计算skip_start、skip_end参数值
            skip_start = (idx != 0) or (skip_start and idx == 0)
            skip_end = (idx != len(text_list) - 1) or (
                skip_end and idx == len(text_list) - 1
            )

            audio = self.infer(
                text=text,
                speaker_name=speaker_name,
                language=language,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                emotion=emotion,
                seed=seed,
                skip_start=skip_start,
                skip_end=skip_end,
                add_blank=add_blank,
            )
            audil_list.append(audio)

        np_audio = np.concatenate(audil_list, axis=2)
        return np_audio


infor_onnx_instance = INFER_ONNX()
