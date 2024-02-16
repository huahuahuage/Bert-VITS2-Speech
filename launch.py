import ctypes
from launch_message import launch_message_instance
ctypes.windll.kernel32.SetConsoleTitleW("花花 Bert-VITS2 原神/星铁语音合成API助手")

# 全文忽略警告信息
import warnings

warnings.filterwarnings("ignore")
import sys
import imp

imp.reload(sys)

print(
    "【程序声明】基于开源项目 Bert-VITS2.1 (https://github.com/fishaudio/Bert-VITS2)。"
)
print(
    "【模型来源】红血球AE3803@bilibili/纳鲁塞缪希娜卡纳@bilibili/原神4.2/星穹铁道1.5/chinese-roberta-wwm-ext-large/deberta-v2-large-japanese/deberta-v3-large。"
)

print("【程序制作】花花花花花歌@bilibili。")

print(
    "【郑重声明】严禁将此软件用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》的用途，严禁将此软件用于任何政治相关用途。\n"
)

from log import log_instance

import gradio
import uvicorn

# 重载uvicorn日志输出格式
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "[%(levelname)s] - %(message)s"
log_config["formatters"]["default"]["fmt"] = "[%(levelname)s] - %(message)s"

# 取读配置文件
from config import config_instance

HOST = config_instance.get("server_host", "127.0.0.1")
PORT = config_instance.get("server_port", 7880)
WEBUI_ENABLE = config_instance.get("webui_enable", True)

log_instance.info("欢迎使用 花花 Bert-VITS2 原神/星铁语音合成API助手。")
log_instance.info(f"程序资源正在载入中，请稍候...")
launch_message_instance.update("资源载入中，请稍候...")


from api.api import app as api_app

if WEBUI_ENABLE:
    from api.ui import app as webui_app

    app = gradio.mount_gradio_app(app=api_app, blocks=webui_app, path="/gradio")
else:
    app = api_app


def run_server():
    """
    程序入口函数
    """
    try:
        uvicorn.run(app=app, host=HOST, port=PORT, log_level="critical")
    except Exception as e:
        log_instance.error("程序启动失败：", e)


if __name__ == "__main__":
    run_server()
    # 用户输入任意键来退出
    log_instance.info("按下任意键退出程序...")
    input()
