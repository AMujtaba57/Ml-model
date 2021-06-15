import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def hair_removal(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    return dst

def resize_img(img):
    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    else:
        img_resize = cv2.resize(img, (224,224))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        return img_resize

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
    coloured[mask == 255] = (0, 0, 255)
    dst = cv2.addWeighted(img, 0.6, coloured, 0.4, 0)
    return dst



def simple_classifier(img):
    model = tf.keras.models.load_model("simpler1.h5", custom_objects=None, compile=True, options=None)
    im_pil = Image.fromarray(img)
    im_np = np.asarray(im_pil)
    img_reshape = im_np[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def resnet_classifier(img):
    model = tf.keras.models.load_model("resnet2.h5")
    im_pil = Image.fromarray(img)
    im_np = np.asarray(im_pil)
    img_reshape = im_np[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction