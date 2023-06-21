#! usr/bin/env/python

import cv2 
import numpy as np 

def threshold(channel, thresh=(128,255), thresh_type = cv2.THRESH_BINARY):

    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

def blur_gaussian(channel, kernel_size=3):

    return cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)

def Perspective(img):

    pass 