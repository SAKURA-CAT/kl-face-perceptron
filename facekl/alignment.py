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
from .detector import mtcnn
import math


def align_face(image: np.ndarray) -> np.ndarray:
    """人脸矫正，需要输入人脸框和人脸关键点，最终将返回裁剪并矫正的人脸图像

    Parameters
    ----------
    image : np.ndarray
        输入图像，在函数内部统一转换为BGR格式
    box : np.ndarray
        人脸框，shape=(1, 4), 分别为x1, y1, x2, y2
    landmarks : np.ndarray
        人脸关键点，shape=(1, 10)
    """
    # 人脸检测
    box, landmarks = mtcnn.detect(image)
    if len(box) == 0 or len(landmarks) == 0:
        raise ValueError("No face detected in the image")
    # 首先裁剪出人脸
    box = box.astype(np.int8)
    image = image[(box[0, 1]) : box[0, 3], box[0, 0] : box[0, 2]]
    # 关键点信息为[n, 10]的浮点数 numpy 数组，包含 x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 的坐标
    # (x1, y1)为左眼坐标，(x2, y2)为右眼坐标，(x3, y3)为鼻尖坐标，(x4, y4)为左嘴角坐标，(x5, y5)为右嘴角坐标

    # 计算人脸倾斜角度
    eye_left = (landmarks[0, 0], landmarks[0, 5])
    eye_right = (landmarks[0, 1], landmarks[0, 6])
    angle = calculate_face_tilt(eye_left, eye_right)
    image = rotate_face(image, angle)
    return image


def calculate_face_tilt(eye_left, eye_right):
    # 计算眼睛之间的斜率
    slope_eyes = (eye_right[1] - eye_left[1]) / (eye_right[0] - eye_left[0])

    # 计算斜率对应的角度
    angle_radians = math.atan(slope_eyes)

    # 将弧度转换为角度
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def rotate_face(image, angle):
    # 获取图像中心点坐标
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 构造旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return rotated_image
