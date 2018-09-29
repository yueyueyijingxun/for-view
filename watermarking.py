#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yijing
# datetime:2018/3/22 15:10
# software: PyCharm

import cv2
import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt

class Components():
    Coefficients = []
    U = None
    S = None
    V = None

class watermarking():
    def __init__(self, watermark_path="C:/Users/xunyijing/Desktop/myqrcode/SVDDWT/half.jpg", ratio=0.05, wavelet="haar",level=1):
        self.level = level
        self.wavelet = wavelet
        self.ratio = ratio
        self.shape_watermark = cv2.imread(watermark_path, 0).shape
        self.W_components = Components()
        self.qrimg_components = Components()
        #self.W_components.Coefficients, self.W_components.U,self.W_components.S, self.W_components.V = self.calculate(watermark_path)
        self.W_components.U, self.W_components.S, self.W_components.V = self.calculate1(watermark_path)
        #self.W_img=cv2.imread(watermark_path,1)
    def calculate(self, img):#计算系数和SVD矩阵，处理图片没有针对性，任何一个图片都可以调用calculate
        if isinstance(img, str):#判断img是否为str类型
            img = cv2.imread(img, 0)
        Coefficients = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)#多层变换，如果图片纹理不够丰富，单层就够了选择的是2，所以返回[cA2,(cH2,cV2,cD2),(cH1,cV1,cD1)]
        self.shape_LL = Coefficients[0].shape#低频分量的维度大小
        U, S, V = np.linalg.svd(Coefficients[0])#使用numpy的线性代数工具箱对QR码低频分量进行SVD分解得到U，S，V三个矩阵，奇异值为S
        return Coefficients, U, S, V#返回了QR图片小波变换后的系数矩阵和低频的奇异值S

    def calculate1(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, 0)
        #Coefficients1 = pywt.wavedec2(img, wavelet=self.wavelet, level=0)
        self.shape_LL = img.shape
        U, S, V = np.linalg.svd(img)
        return U, S, V
    def calculate2(self,img):
        if isinstance(img,str):
            img=cv2.imread(img,0)
        return img

    def diag(self, s):#将奇异值转化为矩阵，一维数组
        S = np.zeros(self.shape_LL)#单位矩阵，大小由d低频分量大小确定
        row = min(S.shape)#row为S的行列最小值，即如果低频分量矩阵S是行列不等时，选小的那个
        S[:row, :row] = np.diag(s)#提取对角线上元素
        return S

    def embed(self):#奇异值相加，嵌入强度ratio=0.1
       self.S_qrimg = self.qrimg_components.S + self.ratio * self.W_components.S*(self.qrimg_components.S.max() / self.W_components.S.max())
       #self.S_qrimg = self.qrimg_components.S + self.ratio * self.W_img

    def recover(self, name):#从svd和dwt中恢复图像
        components = eval("self.{}_components".format(name))# 字符串格式化
        s = eval("self.S_{}".format(name))
        components.Coefficients[0] = components.U.dot(self.diag(s)).dot(components.V)#矩阵相乘
        recover=pywt.waverec2(components.Coefficients, wavelet=self.wavelet)#二阶小波重建，coeffs的形式与wavedec2()的分解结果相同，小波基选择分解时用的小波基
        return recover

    def watermark(self, img="C:/Users/xunyijing/Desktop/myqrcode/picture/shengcheng53.png", path_save=None):#水印主要部分
        if not path_save:#path_save不为None，则执行：后语句
            path_save = "C:/Users/xunyijing/Desktop/myqrcode/picture/watermarkedQRcode515.png"
        self.path_save = path_save
        self.qrimg_components.Coefficients, self.qrimg_components.U,self.qrimg_components.S, self.qrimg_components.V = self.calculate(img)
        self.embed()
        img_rec = self.recover("qrimg")
        cv2.imwrite(path_save, img_rec)

    def extracted(self, image_path=None,extracted_watermark_path = None):#提取水印
        if not extracted_watermark_path:
            extracted_watermark_path = "C:/Users/xunyijing/Desktop/myqrcode/picture/watermark_extracted515.png"
        if not image_path:
            image_path = self.path_save
        img = cv2.imread(image_path,0)
        #img = cv2.resize(img, self.shape_watermark)
        eximg_components = Components()
        eximg_components.Coefficients, eximg_components.U, eximg_components.S, eximg_components.V = self.calculate(img)
        self.S_W = (eximg_components.S - self.qrimg_components.S) / self.ratio
        #watermark_extracted = self.recover("W")
        waterrecover = self.W_components.U.dot(self.diag(self.S_W)).dot(self.W_components.V)  # 矩阵相乘
        #recover = pywt.waverec2(self.W_components.Coefficients, wavelet=self.wavelet)
        cv2.imwrite(extracted_watermark_path, waterrecover)#watermark_extracted)



#计算psnr
def psnr(im1,im2):
    im_array1 = np.array(Image.open(im1))
    im_array2 = np.array(Image.open(im2))
    a,b=np.shape(im_array1)
    diff = np.square(abs(im_array1 - im_array2))
    rmse = np.sqrt(diff.sum()/(a*b))
    psnr = 20*np.log10(255/rmse)
    return psnr

def nc(im1,im2):
    im1 = np.array(Image.open(im1).convert('L'))
    im2 = np.array(Image.open(im2))
    err=np.sum(im1.astype("float")*im2.astype("float"))
    sum1=np.sum(im1.astype("float")*im1.astype("float"))
    sum2=np.sum(im2.astype("float")*im2.astype("float"))
    nc = err / (np.sqrt(sum1) * np.sqrt(sum2))
    return nc
psnr = psnr('C:/Users/xunyijing/Desktop/myqrcode/picture/shengcheng53.png','C:/Users/xunyijing/Desktop/myqrcode/picture/watermarkedQRcode53.png')
nc=nc('C:/Users/xunyijing/Desktop/myqrcode/picture/final.jpg','C:/Users/xunyijing/Desktop/myqrcode/picture/watermark_extracted429.png')
print psnr
print nc

if __name__ == '__main__':
    watermarking = watermarking()
    watermarking.watermark()
    watermarking.extracted()