import sys
import time
import tkinter as tk
from tkinter.font import Font
from tkinter import PhotoImage
import webbrowser
import threading

from config import config_instance
from launch_message import launch_message_instance

HOST = config_instance.get("server_host", "127.0.0.1")
PORT = config_instance.get("server_port", 7880)


# 创建一个窗口
window = tk.Tk()

# 设置窗口图标
window.wm_iconbitmap("icon.ico")

# 设置窗口标题
window.title("花花原神/星铁语音合成 --power by bert-vits2 2.1")

# 设置窗口大小
window.geometry("750x450")

# 加载logo图片
image_file = "icon.png"  # 替换为你的图片路径
image = PhotoImage(file=image_file)

# 创建一个Frame作为容器
frame = tk.Frame(window)

# 图片
image_label = tk.Label(frame, image=image)
image_label.pack(side="top", fill="x")

# 文本
# 定义一个字体对象，指定字体名称、大小和样式
# 字体名称可以是系统支持的任何字体，大小是字体的大小，样式可以是'normal', 'bold', 'italic' 或它们的组合
text_font = Font(family="微软雅黑", size=22, weight="normal")
text_label = tk.Label(frame, text="正在启动中...", font=text_font)
text_label.pack(side="top", fill="x")  # 将文本标签放在Frame的顶部，并填充水平空间


# 定义一个函数，该函数将在按钮被点击时调用
def button_click():
    url = f"http://{HOST}:{PORT}/gradio"
    # 打开默认的浏览器并访问指定的URL
    webbrowser.open(url)


# 按钮
button = tk.Button(
    frame, text=f"打开WEB控制台", command=button_click, highlightthickness=5
)


# 定义一个函数来计算并设置Label的位置，以便图片居中
def set_label_position(event=None):
    # 获取窗口的宽度和高度
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # 获取图片（Label）的宽度和高度
    image_width = frame.winfo_reqwidth()
    image_height = frame.winfo_reqheight()

    # 计算x和y坐标以居中图片
    x_coordinate = (window_width - image_width) // 2
    y_coordinate = (window_height - image_height) // 2

    # 使用place方法将Label放置在计算出的坐标上
    frame.place(x=x_coordinate, y=y_coordinate, anchor="nw")


# 定义一个函数来更新Label的文本内容
def update_label_text(text: str):
    text_label.config(text=text)


# 定义一个函数来更新后台运行信息
def run_message_in_background():
    """
    后台运行信息更新
    """

    def __task():
        while True:
            launch_message = launch_message_instance.get()
            if launch_message != "":
                update_label_text(launch_message)
            if launch_message == "程序资源载入已完成":
                button.pack(side="top", pady=10)
                return
            time.sleep(1)

    t = threading.Thread(target=__task)
    t.start()


# 定义一个函数来创建后台线程
def run_server_in_background():
    """
    创建后台线程
    """

    def __task():
        from launch import run_server

        run_server()

    # 创建一个后台线程
    t = threading.Thread(target=__task)
    t.start()


# 调用函数以设置初始位置
set_label_position()

# 如果窗口大小改变，重新计算Label的位置
window.bind("<Configure>", set_label_position)

# 将Label添加到窗口（但使用place方法已经在set_label_position中完成了）
# 注意：不再需要调用pack、grid或place方法，因为已经在set_label_position中调用了place

# 后台运行信息更新
run_message_in_background()
run_server_in_background()

# 运行窗口，直到用户关闭它
window.mainloop()
sys.exit()