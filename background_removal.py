import cv2 
import pixellib
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from pixellib.tune_bg import alter_bg

def read_img(img_data, img_name):
    if img_data != '':
        cv2.imwrite(img_name, img_data)
def back_removal(img):
    change_bg = alter_bg()
    change_bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
    output = change_bg.color_bg(img, colors = (0, 128, 0))
    return output

def segmentation(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh =     cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    kernel = np.ones((6,6),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    blur = cv2.blur(dilate,(15,15))
    ret, thresh =     cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy =     cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)
    coloured = res.copy()
    coloured[mask == 255] = (0, 100, 255)
    dst = cv2.addWeighted(img, 0.6, coloured, 0.4, 0)
    return dst

def contoursDraw(img):
    print(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    img = np.uint8(img)
    edged = cv2.Canny(img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, 0, (0, 130, 0), 5)
    cv2.drawContours(img, contours, 2, (0, 155, 0), 3)
    return img

def sobel_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = filters.sobel(gray)
    blurred = filters.gaussian(sobel, sigma=2.0)
    return blurred