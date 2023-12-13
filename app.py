#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-12 16:00:56
@File: app.py
@IDE: vscode
@Description:
    展示程序，基于gradio实现可视化展示
"""
from facekl import detect
import gradio as gr
import os


def predict(image):
    """
    :param image: input image
    """
    _, image = detect(image, None)
    return image


if __name__ == "__main__":
    # 测试图像为image中名称包含image的图像
    examples = ["imgs/Jim_OBrien_0003.jpg", "imgs/Martha_Bowen_0002.jpg"]
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Image"),
        outputs=[gr.Image(type="pil", label="Compose Image")],
        title="kl-face-perceptron",
        examples=examples,
        theme=gr.themes.Base(),
    )
    interface.launch()
