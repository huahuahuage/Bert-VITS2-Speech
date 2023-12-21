import json
import chardet
import logging

CONFIG_PATH = "config.json"


def read_config(config_path:str) -> dict:
    """
    取读配置文件
    """
    f = open(config_path, "rb")
    try:
        raw_data:str = f.read()
        # 检测配置文件编码
        char_type = chardet.detect(raw_data)['encoding']
        # 解码
        data = raw_data.decode(char_type)
        config_data = json.loads(data)
    except:
        config_data = {}
        logging.error(f"配置文件 {config_path} 不存在或者格式错误。")

    f.close()

    return config_data


class ONNX_CONFIG:
    """
    配置文件
    """

    def __init__(self) -> None:
        logging.info(f"正在加载配置文件...")
        self.config_data = read_config(CONFIG_PATH)

    def get(self, key, default=None):
        """
        获取配置信息
        """
        return self.config_data.get(key, default)

config_instance = ONNX_CONFIG()
