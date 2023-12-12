#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-12 15:59:50
@File: test/face_detect.py
@IDE: vscode
@Description:
    检测文件，此处用于测试检测效果
"""
from facekl import detect
import cv2

target = cv2.imread("imgs/Jim_OBrien_0003.jpg")
sources = [
    "imgs/Jim_OBrien_0001.jpg",
    "imgs/Martha_Bowen_0001.jpg",
    "imgs/Janica_Kostelic_0001.jpg",
    "imgs/Elisabeth_Schumacher_0001.jpg",
    "imgs/Debra_Messing_0001.jpg",
]
res = detect(target, sources)
print(res)
