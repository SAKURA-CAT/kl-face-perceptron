#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-07 22:47:25
@File: facekl\face_alignment.py
@IDE: vscode
@Description:
    本模块用于人脸对齐，基于MTCNN模型实现
    人脸对齐在facekl中并不是必须的，可以选择性使用它
"""
import cv2
import numpy as np
from typing import Tuple


def align_face(image: np.ndarray) -> np.ndarray:
    """人脸对齐，需要输入人脸框和人脸关键点，最终将返回裁剪并对齐的人脸图像

    Parameters
    ----------
    image : np.ndarray
        输入图像，在函数内部统一转换为BGR格式
    box : np.ndarray
        人脸框，shape=(1, 4), 分别为x1, y1, x2, y2
    landmarks : np.ndarray
        人脸关键点，shape=(1, 10)
    """
    box, landmarks : Tuple[np.ndarray, np.ndarray] = detect_face(image)
    if len(box) == 0 or len(landmarks) == 0:
        raise ValueError("No face detected in the image")
    # 在此处完成人脸对齐的过程


def detect_face(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """人脸检测，基于MTCNN模型实现

    Parameters
    ----------
    image : np.ndarray
        输入图像
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        人脸框和人脸关键点，shape=(1, 4)和shape=(1, 10)
    """
    # 这里采用动态导入的方式，添加提示信息
    try:
        from mtcnnruntime import MTCNN
    except ImportError:
        error = "If you need to use face alignment, you need to install the mtcnnruntime module first: pip install mtcnn-runtime"
        raise ImportError(error)

    mtcnn = MTCNN()
    
    
    return mtcnn.detect(image)
