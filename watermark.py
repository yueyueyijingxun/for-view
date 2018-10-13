#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yijing
# datetime:2018-10-13 10:29
# software: PyCharm

import cv2
import pywt
import numpy as np
from PIL import Image
import pytesseract

class Components():
    Coefficients = []
    U = None
    S = None
    V = None

class extract():
    def __init__(self,watermark_path=None, qrimg_path=None, ratio=0.05, wavelet="haar",level=1):
        watermark_path = "C:/Users/xunyijing/Desktop/half.jpg"#原始水印图像
        qrimg_path="C:/Users/xunyijing/Desktop/shengcheng53.png"#原始QR码图像
        self.level = level
        self.wavelet = wavelet
        self.ratio = ratio
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        self.W_components = Components()
        self.qrimg_components = Components()
        self.W_components.U, self.W_components.S, self.W_components.V = self.calculate1(watermark_path)
        self.qrimg_components.Coefficients, self.qrimg_components.U, self.qrimg_components.S, self.qrimg_components.V = self.calculate(qrimg_path)

    def calculate(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, 0)
            Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)
            self.shape_LL = Coefficients[0].shape
            U, S, V = np.linalg.svd(Coefficients[0])
            return Coefficients, U, S, V

    def calculate1(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        U, S, V = np.linalg.svd(img)
        return U, S, V

    def diag(self, s):
        S = np.zeros(self.shape_LL)
        row = min(S.shape)
        S[:row, :row] = np.diag(s)
        return S

    def extracted(self, image_path=None, extracted_watermark_path=None):  # 提取水印
        if not extracted_watermark_path:
            extracted_watermark_path = "C:/Users/xunyijing/Desktop/watermarkedimg.png"#存提取图片的路径
        if not image_path:
            image_path = "C:/Users/xunyijing/Desktop/watermarkedQRcode53.png"#含水印的二维码图像
        C,U,S,V=self.calculate(image_path)
        S_W = (S - self.qrimg_components.S) / self.ratio
        a = self.diag(S_W)
        waterrecover = self.W_components.U.dot(a).dot(self.W_components.V)  # 矩阵相乘，SVD重构
        cv2.imwrite(extracted_watermark_path, waterrecover)

if __name__ == '__main__':
    extract=extract()
    extract.extracted()