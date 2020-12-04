#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:14:45 2020

@author: ratul
"""
import cv2, argparse, os
import numpy as np
import matplotlib.pyplot as plt


rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299,0.587,0.114])

def dither(num, thresh = 127):
    
    derr = np.zeros(num.shape, dtype=float)

    div = 8
    for y in range(num.shape[0]):
        for x in range(num.shape[1]):
            newval = derr[y,x] + num[y,x]
            if newval >= thresh:
                errval = newval - 255
                num[y,x] = 1.
            else:
                errval = newval
                num[y,x] = 0.
            if x + 1 < num.shape[1]:
                derr[y, x + 1] += errval / div
                if x + 2 < num.shape[1]:
                    derr[y, x + 2] += errval / div
            if y + 1 < num.shape[0]:
                derr[y + 1, x - 1] += errval / div
                derr[y + 1, x] += errval / div
                if y + 2< num.shape[0]:
                    derr[y + 2, x] += errval / div
                if x + 1 < num.shape[1]:
                    derr[y + 1, x + 1] += errval / div
    return num[::-1,:] * 255

def plot_image(img, dithered_img):
    plt.subplot(211)
    plt.imshow(img)
    plt.title('Image before preprocessing')
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(dithered_img[::-1,:])
    plt.title('After processing %d'%sum(dithered_img.flatten()==0))
    plt.axis('off')
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='generated-imgs/brown_hair_heavy_makeup_attractive_no_beard_young_eyeglasses_smiling.jpg', type=str, help='weight path of generator')
    args = parser.parse_args()
    
    os.makedirs('dithered-imgs', exist_ok=True)
    
    # read gray image
    img = cv2.imread(args.path,0)
    dithered_img = dither(img.copy(), 128)
    
    plot_image(img, dithered_img)
    cv2.imwrite(f'dithered-imgs/dithered_{os.path.basename(args.path)}', dithered_img[::-1,:])
    
    