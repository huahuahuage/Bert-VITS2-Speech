"""
这里实现Bert Tokenizer的加载
"""

from log import log_instance
from dataclasses import dataclass
from transformers import BertTokenizer, DebertaV2TokenizerFast
from config import config_instance

from api.utils import os_type_instance


CHINESE_ONNX_LOCAL_DIR = config_instance.get("bert_chinese", "")
JAPANESE_ONNX_LOCAL_DIR = config_instance.get("bert_japanese", "")
ENGLISH_ONNX_LOCAL_DIR = config_instance.get("bert_english", "")

if os_type_instance.type == "Windows":

    @dataclass
    class BertTokenizerDict:
        """
        ONNX分析器对象字典 for windows
        """

        ZH: BertTokenizer = (
            BertTokenizer.from_pretrained(CHINESE_ONNX_LOCAL_DIR)
            if CHINESE_ONNX_LOCAL_DIR
            else None
        )
        JP: DebertaV2TokenizerFast = (
            DebertaV2TokenizerFast.from_pretrained(JAPANESE_ONNX_LOCAL_DIR)
            if JAPANESE_ONNX_LOCAL_DIR
            else None
        )
        EN: DebertaV2TokenizerFast = (
            DebertaV2TokenizerFast.from_pretrained(ENGLISH_ONNX_LOCAL_DIR)
            if ENGLISH_ONNX_LOCAL_DIR
            else None
        )

else:

    @dataclass
    class BertTokenizerDict:
        """
        ONNX分析器对象字典 for linux
        """

        ZH: BertTokenizer = (
            BertTokenizer.from_pretrained(CHINESE_ONNX_LOCAL_DIR)
            if CHINESE_ONNX_LOCAL_DIR
            else None
        )
        JP: BertTokenizer = (
            BertTokenizer.from_pretrained(JAPANESE_ONNX_LOCAL_DIR)
            if JAPANESE_ONNX_LOCAL_DIR
            else None
        )
        EN: DebertaV2TokenizerFast = (
            DebertaV2TokenizerFast.from_pretrained(ENGLISH_ONNX_LOCAL_DIR)
            if ENGLISH_ONNX_LOCAL_DIR
            else None
        )


log_instance.info("正在加载BERT分析器...")
tokenizer_instance = BertTokenizerDict()
