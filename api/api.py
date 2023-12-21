import time

time_for_launch_start = time.time()

import os
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from .tts import tts_instance

# 取读配置文件
from log import log_instance
from config import config_instance

HOST = config_instance.get("server_host", "127.0.0.1")
PORT = config_instance.get("server_port", 7880)
WEBUI_ENABLE = config_instance.get("webui_enable", True)


def handle_startup_event():
    # 后台线程启动webui
    time_for_launch_end = time.time()
    log_instance.info(
        f"程序资源载入已完成，耗时：{str(time_for_launch_end - time_for_launch_start)}\n"
    )
    api_string_zh = (
        f"http://{HOST}:{PORT}/api/tts?speaker=珊瑚宫心海&text="
        + "{text}"
        + "&format=wav&language=auto&length=1&sdp=0.4&noise=0.6&noisew=0.8&emotion=7&seed=114514"
    )

    if WEBUI_ENABLE:
        log_instance.info(f"网页控制台 -> http://{HOST}:{PORT}/gradio")

    log_instance.info(f"接口地址 -> {api_string_zh}")
    log_instance.info("请求方式 -> GET")
    log_instance.info("文件类型 -> wav\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    fastapi生命周期函数
    """
    # 启动事件
    handle_startup_event()
    yield
    # 结束事件


app = FastAPI(lifespan=lifespan)


@app.get("/api/tts")
def get_data(
    text: str,
    speaker: str,
    language: Optional[str] = "ZH",
    sdp: Optional[float] = 0.2,
    noise: Optional[float] = 0.6,
    noisew: Optional[float] = 0.8,
    length: Optional[float] = 1,
    emotion: Optional[int] = 7,
    seed: Optional[int] = 114514,
):
    log_string = f"收到文本转语音请求：{speaker} -> {text}"
    log_error_string = f"文本转语音推理失败：{speaker} -> {text}"

    try:
        start = time.time()
        file_path = tts_instance.gen_tts(
            text=text,
            speaker_name=speaker,
            language=language,
            sdp_ratio=sdp,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            emotion=emotion,
            seed=seed,
        )
        stop = time.time()
        log_instance.info(f"{log_string} 耗时：{str(stop - start)}")
    except Exception as e:
        log_instance.error(f"{log_error_string} {str(e)}")
        return {"code": -1, "data": f"{str(e)}。"}

    if not file_path:
        log_instance.error(f"{log_error_string} 请检查请求参数是否正确。")
        return {"code": -1, "data": "语音生成失败，请检查请求参数是否正确。"}

    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        log_instance.error(f"{log_error_string} 语音文件读取失败。")
        return {"code": -2, "data": "数据读取失败，请重试。"}
    except:
        log_instance.error(f"{log_error_string} 请检查请求参数是否正确。")
        return {"code": -1, "data": "语音生成失败，请检查请求参数是否正确。"}

    return FileResponse(
        file_path,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        media_type="audio/basic",
        filename=os.path.basename(file_path),
    )
