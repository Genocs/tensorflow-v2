# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:00:00 2020

@author:       Genocs
@description:  The hello world for image analysis
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Import numpy
import numpy as np

# Import OpenCV
import cv2

# Import matplotlib lib
import matplotlib.pyplot as plt

# Import Tesseracts
import pytesseract

# Import utility libraries
import datetime
import time


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image,
                             M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


t1 = time.time()

img = cv2.imread('.\\data\\image.jpg')

gray = get_grayscale(deskew(img))
# thresh = thresholding(gray)
# rnoise = remove_noise(thresh)
# dilate = dilate(thresh)
# erode = erode(gray)
# opening = opening(gray)
# canny = canny(thresh)

h, w = gray.shape

# Adding custom options
custom_config = r'--oem 1 -c tessedit_char_whitelist=<ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 --psm 6'

boxes = pytesseract.image_to_boxes(gray, config=custom_config)
result = pytesseract.image_to_string(img, config=custom_config)

t2 = time.time()

# get the difference as datetime.timedelta object
diff = (datetime.datetime.fromtimestamp(t1) - datetime.datetime.fromtimestamp(t2))
print('Result is: %s' % result)
# diff is negative as t2 is in the future compared to t2
print('difference is {0} seconds'.format(abs(diff.total_seconds())))

for b in boxes.splitlines():
    b = b.split(' ')
    gray = cv2.rectangle(gray, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 255, 255), 1)

# for b in boxes.splitlines():
#    b = b.split(' ')
#    # text 
#    text = b[0]
#  
#    # font 
#    font = cv2.FONT_HERSHEY_SIMPLEX 
#  
#    # org 
#    org = (int(b[1]), h - int(b[2])) 
#  
#    # fontScale
#    fontScale = 1
#   
#    # Red color in BGR 
#    color = (255, 255, 255) 
#  
#    # Line thickness of 2 px 
#    thickness = 1
#    gray = cv2.putText(gray, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 


# Use the matplotlib to sho the first trainingset sample
plt.figure()
plt.imshow(gray)
plt.colorbar()
plt.grid(False)
plt.show()

# cv2.imshow('img', canny)
# images = [gray, thresh, rnoise, canny]
# show_images(images, 3, ["gray", "thresh", "rnoise", "canny"])
