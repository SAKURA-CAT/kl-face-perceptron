#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-12 15:51:33
@File: facekl/main.py
@IDE: vscode
@Description:
    facekl主程序
"""
from typing import List
import numpy as np
from .alignment import align_face
import cv2
import os

# 当前运行时的路径
DATASET = os.path.join(os.getcwd(), "dataset.npz")


def detect(target: np.ndarray, sources: List[str], size: int = 64) -> dict:
    """检测函数，用于检测目标图片中是否存在源图片中的人脸并返回检测结果

    Parameters
    ----------
    target : np.ndarray
        检测目标图片
    sources : List[np.ndarray]
        检测源图片列表，可以是多张图片，程序检测是否已保存dataset，如果已保存，这个参数没有意义
    size : int, optional
        图像标准化尺寸，默认为256

    Returns
    -------
    dict
        检测结果，包含`id`和`scores`两个字段，`id`为检测结果的`index`，`scores`为实际的置信度列表
    """
    # target标准化
    target = standardize(target, size)
    c, vc, mean, sources = create_dataset(sources, size)
    # 计算target的系数
    t = vc.T.dot(target - mean)
    # 依次计算sources中每个人脸与target的距离，取距离最小的作为检测结果
    scores = []
    for i in range(c.shape[1]):
        scores.append(np.linalg.norm(t - c[:, i]))
    idx = np.argmin(scores)
    print("this is", sources[idx], "with score", scores[idx])

    return {"id": idx, "scores": scores}, cv2.cvtColor(cv2.imread(sources[idx]), cv2.COLOR_BGR2RGB)


def create_dataset(sources: List[str], size: int = 64, threshold: float = 0.99) -> np.ndarray:
    """根据给定的路径创建数据集

    Parameters
    ----------
    sources: List[str]
        数据集路径，要求数据集中的每个文件夹中包含一个人的所有照片
    size : int, optional
        图像标准化尺寸，默认为256

    Returns
    -------
    np.ndarray
        数据集，shape=(n, size*size)
    """
    # 读取数据集
    if os.path.exists(DATASET):
        dataset = np.load(DATASET)
        return dataset["c"], dataset["vc"], dataset["mean"], dataset["sources"]
    # 创建数据集
    ss = [cv2.imread(i) for i in sources]
    # 标准化
    ss = [standardize(s, size) for s in ss]
    # 将sources中每个列向量合并得到一个矩阵，size*size的图像转换为(size*size)*len(sources)的矩阵
    ss = np.array(ss).T
    # 计算sources的协方差矩阵
    cov = np.cov(ss)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # 对特征值进行排序, 从大到小, 同时排序特征向量
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # 计算特征值的和，取前k个特征值的和大于总和的阈值
    total = sum(eigenvalues)
    k = 0
    for i, _ in enumerate(eigenvalues):
        if sum(eigenvalues[:i]) / total >= threshold:
            k = i
            break
    # 取前k个特征向量，得到vc
    vc = eigenvectors[:, idx[:k]]
    # 最终的系数矩阵为vc.T * (ss - mean)
    c = vc.T.dot(ss - np.mean(ss, axis=1).reshape(-1, 1))
    # 保存数据集，顺便保存数据源
    np.savez(DATASET, c=c, vc=vc, mean=np.mean(ss, axis=1), sources=sources)
    return c, vc, np.mean(ss, axis=1), sources


def standardize(img: np.ndarray, size: int):
    """标准化一个图像,裁切、缩放、灰度化

    Parameters
    ----------
    img : np.ndarray
        图像
    size : int
        图像标准化尺寸
    """
    # target人像矫正, 裁切出人脸
    img = align_face(img)
    # 图像灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像标准化，将图像转换为size*size的大小
    img = cv2.resize(img, (size, size))
    # 图像向量化，将图像转换为一维列向量
    img = img.flatten()
    return img
