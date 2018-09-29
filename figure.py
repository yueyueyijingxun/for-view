#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yijing
# datetime:2018/4/29 18:07
# software: PyCharm

import cv2
import pywt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#自己实现
#读取qr图像，进行dwt和svd
qrimg=np.array(Image.open("C:/Users/xunyijing/Desktop/myqrcode/picture/shengcheng424.png"))
qrCoefficients = pywt.wavedec2(qrimg, wavelet="haar", level=1)#DWT分解，选择小波基为haar，二级变换
shape_LL = qrCoefficients[0].shape
S = np.zeros(shape_LL)
row = min(S.shape)
Uqr, Sqr, Vqr = np.linalg.svd(qrCoefficients[0])#qr低频SVD
Sqr= np.diag(Sqr)
#读取水印图像
waterimg=np.array(Image.open("C:/Users/xunyijing/Desktop/myqrcode/picture/half.jpg"))
Uw,Sw,Vw=np.linalg.svd(waterimg)#水印图像SVD
Sw=np.diag(Sw[:,1])
ratio=0.05
T=Sqr+ratio*Sw#水印嵌入，强度选择ratio=0.05
Ut,St,Vt=np.linalg.svd(T)#对T再进行svd
St=np.diag(St)
cA1=Uqr.dot(St).dot(Vqr)#奇异值重构
qrnew=[cA1,qrCoefficients[1]]
waterqr=pywt.waverec2(qrnew, wavelet="haar")
a=Image.fromarray(waterqr)
if a.mode != 'RGB':
    a = a.convert('RGB')
a.save("C:/Users/xunyijing/Desktop/myqrcode/picture/430qrwater.jpg")
#cv2.imwrite('C:/Users/xunyijing/Desktop/myqrcode/picture/430qrwater.jpg',waterqr,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
#cv2.imwrite('C:/Users/xunyijing/Desktop/myqrcode/picture/430qrwater.png',waterqr,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imshow('img',waterqr)
cv2.waitKey(0)






