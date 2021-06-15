from preprocessing import resize_img, hair_removal, simple_classifier, segmentation, resnet_classifier
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import urllib
from background_removal import back_removal, segmentation, contoursDraw, sobel_img, read_img

st.title("SkinCare Recommender System")

st.header("""
    Choose an Image :
""")
uploaded_file = st.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    bytes_data = uploaded_file.read()
    image = Image.open(BytesIO(bytes_data))
    image = np.array(image)
    open_cv_image = image[:, :, ::-1].copy() 

    st.write(uploaded_file.name)

    read_img(open_cv_image, uploaded_file.name)
    br = back_removal(uploaded_file.name)
    

    sobel = sobel_img(open_cv_image)
    cnt = contoursDraw(open_cv_image)
    img_seg = segmentation(open_cv_image)

    col1, col2 = st.beta_columns(2)
    col1.write("Original Image")
    col1.image(uploaded_file)

    col2.write("Background Remove Image")
    col2.image(br)

    col3, col4, col5 = st.beta_columns(3)
    col3.write("Blurred Image")
    col3.image(sobel)

    col4.write("Contoured Image")
    col4.image(cnt)

    col5.write("Final Segmentized Image")
    col5.image(img_seg)