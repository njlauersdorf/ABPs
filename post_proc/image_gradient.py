#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:51:48 2021

@author: nicklauersdorf
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

ddepth = cv2.CV_16S
kernel_size = 5
window_name = "Laplace Demo"
    
#img = cv2.imread('/Volumes/External/test_video_mono/density_pa150_pb500_xa50_ep1.0_phi60_pNum100000_frame_0488.png',0)
imgPath='/Volumes/External/test_video_mono/'
imageName = 'density_pa450_pb500_xa50_ep1.0_phi60_pNum100000_frame_0488.png'
src = cv2.imread(cv2.samples.findFile(imgPath+imageName), cv2.IMREAD_COLOR)
    
#src = cv2.GaussianBlur(src, (5, 5), 0)

#cv2.imwrite(imgPath+'grad2_'+imageName, src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Create Window
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
abs_dst = cv2.convertScaleAbs(dst)
    # [convert]
    # [display]
cv2.imshow(window_name, abs_dst)
cv2.imwrite(imgPath+'grad_'+imageName, abs_dst)
#cv2.waitKey(0)

#plt.savefig(imgPath + 'grad_' +imageName, dpi=200)
stop
    

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'Greys')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'Greys')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'Greys')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'Greys')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()