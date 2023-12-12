#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-07 14:24:26
@File: test/test_mtcnn.py
@IDE: vscode
@Description:
    测试文件, 在此处测试mtcnn的效果
"""
import cv2
from mtcnnruntime import MTCNN


mtcnn = MTCNN()

img = cv2.imread("test/imgs/Elisabeth_Schumacher_0001.jpg")


print(mtcnn.detect(img))
