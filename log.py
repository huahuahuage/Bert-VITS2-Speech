import logging
import colorlog
import jieba

# 调整jieba日志输出级别
jieba.setLogLevel(jieba.logging.ERROR)


# 禁止某些模块日志输出
DISABLED_LOGGER = ["gradio.processing_utils", "gradio", "httpx"]

for logger_name in DISABLED_LOGGER:
    logger_object = logging.getLogger(logger_name)
    logger_object.setLevel(logging.ERROR)

# 创建新的logger
log_instance = logging.getLogger()
log_instance.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_formatter = colorlog.ColoredFormatter(
    fmt="[%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "white",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(console_formatter)

if not log_instance.handlers:
    log_instance.addHandler(console_handler)
