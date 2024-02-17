import os
import time
import platform
import threading
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


def clean_auto_loop(dir_path, interval: int = 0):
    """
    自动清理缓存
    """
    if interval == 0:
        return

    def __task():
        while True:
            time.sleep(interval * 60)
            # 获取当前目录下所有文件
            files = os.listdir(dir_path)
            # 逐个删除文件
            for file in files:
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"文件 {file} 已删除")
                except OSError as e:
                    print(f"删除文件 {file} 时发生错误: {e.strerror}")

    t = threading.Thread(target=__task)
    t.start()


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
