import os
import platform
import shutil
import pyperclip
from log import log_instance


def rebuild_temp_dir(dir_path: str, tips: str = "正在清空API接口语音缓存..."):
    """
    清空重建缓存文件夹
    """
    # 删除缓存
    try:
        shutil.rmtree(dir_path)
        log_instance.info(tips)
    except OSError as e:
        pass
    # 重新建立缓存文件夹
    os.makedirs(dir_path, exist_ok=True)


def copy_to_clipboard(text: str):
    """
    复制字符串到剪切板
    """
    pyperclip.copy(text)


class OSType:
    def __init__(self) -> None:
        self.type = self.check_os()

    def check_os():
        """
        检查操作系统类型
        """
        system = platform.system()
        if system == "Windows":
            return "Windows"
        elif system == "Linux":
            return "Linux"
        else:
            return "MacOS"


os_type_instance = OSType()