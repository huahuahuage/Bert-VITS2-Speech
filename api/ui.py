import os
import time
from log import log_instance
import gradio as gr

from .tts import tts_instance
from .utils import rebuild_temp_dir, copy_to_clipboard
from .utils import os_type_instance

from config import config_instance

HOST = config_instance.get("server_host", "127.0.0.1")
PORT = config_instance.get("server_port", 7880)

SPEAKERS_LIST = tts_instance.get_speakers_list()
LANGUAGES_LIST = ["自动识别", "仅中文", "仅英文", "仅日文"]
LANGUAGES_DICT = {"自动识别": "auto", "仅中文": "ZH", "仅英文": "EN", "仅日文": "JP"}


def __handle_speaker_name(speaker_name):
    """
    处理成接口可识别的发音人名字
    该方法根据模型角色名不同，自行自定义
    """
    split_mark = "["
    if split_mark in speaker_name:
        return "".join(speaker_name.split(split_mark)[:-1])
    return speaker_name


def __copy_url(
    speaker: str,
    language: str = "ZH",
    sdp: float = 0.2,
    noise: float = 0.6,
    noisew: float = 0.8,
    length: float = 1,
    emotion: int = 7,
    seed: int = 114514,
):
    """
    # 将接口地址发送到剪切板
    """
    language = LANGUAGES_DICT[language]
    # 重命名角色名
    speaker = __handle_speaker_name(speaker)

    url = (
        f"http://{HOST}:{PORT}/api/tts?speaker={speaker}&text="
        + "{text}"
        + f"&format=wav&language={language}&length={str(length)}&sdp={str(sdp)}&noise={str(noise)}&noisew={str(noisew)}&emotion={str(emotion)}&seed={str(seed)}"
    )

    copy_to_clipboard(url)

    gr.Info(f"接口地址已复制到剪切板。")
    gr.Info(f"文件类型：wav")
    gr.Info(f"请求方式：GET")
    gr.Info(f"接口地址：{url}")


def __tts(
    text: str,
    speaker: str,
    language: str = "ZH",
    sdp: float = 0.2,
    noise: float = 0.6,
    noisew: float = 0.8,
    length: float = 1,
    emotion: int = 7,
    seed: int = 114514,
    within_interval:float = 0.5,
    sentence_interval: float = 1.0,
    paragraph_interval: float = 2.0,
):
    log_string = f"收到文本转语音请求：{speaker} -> {text}"
    log_error_string = f"文本转语音推理失败：{speaker} -> {text}"

    language = LANGUAGES_DICT[language]
    # 重命名角色名
    speaker = __handle_speaker_name(speaker)

    # try:
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
        within_interval=within_interval,
        sentence_interval=sentence_interval,
        paragraph_interval=paragraph_interval,
    )
    stop = time.time()
    log_instance.info(f"{log_string} 耗时：{str(stop - start)}")
    # except Exception as e:
    #     log_instance.error(f"{log_error_string} {str(e)}")
    #     gr.Error(f"{log_error_string} {str(e)}")
    #     return f"文本转换语音失败，{str(e)}", None
    return "文本转换语音成功", file_path


def webui():
    """
    webui界面
    """
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("欢迎使用 花花 Bert-VITS2 原神/星铁语音合成API助手")
        input_text = gr.TextArea(label="文本内容", placeholder="请输入需要转换语音的文本内容。")
        with gr.Row():
            select_speaker = gr.Dropdown(choices=SPEAKERS_LIST, value=0, label="发音人")
            select_language = gr.Dropdown(choices=LANGUAGES_LIST, value=0, label="语言")
            with gr.Column():
                button_generate = gr.Button(value="生成", variant="primary")
                button_api = gr.Button(value="获取API地址", variant="primary")
        # 输出
        output_status = gr.Textbox(label="状态信息")
        output_audio = gr.Audio(label="输出音频")

        # 其他参数
        with gr.Accordion(label="更多参数配置", open=False):
            slider_emotion = gr.Slider(
                minimum=0,
                maximum=9,
                value=7,
                step=1,
                interactive=True,
                label="情感数值 Emotion",
            )
            slider_sdp_ratio = gr.Slider(
                minimum=0.1,
                maximum=1,
                value=0.2,
                step=0.1,
                interactive=True,
                label="语音语调 SDP Ratio",
            )
            slider_noise_scale_w = gr.Slider(
                minimum=0.1,
                maximum=2,
                value=0.8,
                step=0.1,
                interactive=True,
                label="语音速度 Noise_W",
            )
            slider_noise_scale = gr.Slider(
                minimum=0.1,
                maximum=2,
                value=0.6,
                step=0.1,
                interactive=True,
                label="感情变化 Noise",
            )
            slider_length_scale = gr.Slider(
                minimum=0.1,
                maximum=2,
                value=1.0,
                step=0.1,
                interactive=True,
                label="音节长度 Length",
            )
            slider_seed = gr.Slider(
                minimum=0,
                maximum=1000000,
                value=114514,
                step=1000,
                interactive=True,
                randomize=True,
                label="随机种子 Seed",
            )
            within_interval = gr.Slider(
                minimum=0,
                maximum=10.0,
                value=0.5,
                step=0.1,
                label="句内停顿(秒) ",
            )
            sentence_interval = gr.Slider(
                minimum=0,
                maximum=10.0,
                value=1.0,
                step=0.1,
                label="句间停顿(秒) ",
            )
            paragraph_interval = gr.Slider(
                minimum=0,
                maximum=10.0,
                value=2.0,
                step=0.1,
                label="段间停顿(秒) ",
            )

        # 生成按钮事件
        button_generate.click(
            __tts,
            inputs=[
                input_text,
                select_speaker,
                select_language,
                slider_sdp_ratio,
                slider_noise_scale,
                slider_noise_scale_w,
                slider_length_scale,
                slider_emotion,
                slider_seed,
                within_interval,
                sentence_interval,
                paragraph_interval,
            ],
            outputs=[output_status, output_audio],
        )
        # API按钮事件
        button_api.click(
            __copy_url,
            inputs=[
                select_speaker,
                select_language,
                slider_sdp_ratio,
                slider_noise_scale,
                slider_noise_scale_w,
                slider_length_scale,
                slider_emotion,
                slider_seed,
            ],
        )

    # 重新建立缓存文件夹(for windows)
    if os_type_instance.type == "Windows":
        temp_path = os.path.join(os.path.dirname(os.getenv("APPDATA")), "Temp/gradio/")
        rebuild_temp_dir(temp_path, tips="正在清空Gradio组件语音缓存...")

    return app


app = webui()
