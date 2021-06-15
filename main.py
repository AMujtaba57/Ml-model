from preprocessing import resize_img, hair_removal, simple_classifier, segmentation, resnet_classifier
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import urllib


st.title("SkinCare Recommender System")
st.write("""
    ## Working of SCRS
    ### Which One is Best Model! 
    #### Based on 3 Dataset's classes
""")

st.sidebar.header("""
    Choose an Image :
""")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)

st.sidebar.header("""
    Select Architecture Model :
""")
selective_arch = st.sidebar.selectbox("Select Architcture: ", ("Simpler Method", "Vgg16", "Resnet-50", "EffecientNet", "Inception"))
st.write("You have selected "+selective_arch+" To check The Accuracy of Model")


st.write("""
        ### Step-I
        #### Uploaded Image:
    """)
if uploaded_file:
    st.image(uploaded_file)
    bytes_data = uploaded_file.read()
    st.write(BytesIO(bytes_data))
    image = Image.open(BytesIO(bytes_data))
    image = np.array(image)
    open_cv_image = image[:, :, ::-1].copy() 
    st.write("""

        ## Step-II
        ### Preprocessing:
        #### Resize Image:

    """)
    img = resize_img(open_cv_image)
    st.write(""" Image is resized to """)
    st.write(img.shape)
    st.image(img)
    st.write("""
        
        #### Noise Remove:

    """)
    img = hair_removal(img)
    st.image(img)
    st.write("""
        #### Semantic Segmentation:
         Effected part is highlighted
    """)
    img_seg = segmentation(img)
    st.image(img_seg)
    st.write("""
        ##### Prediction:
    """)
    if selective_arch == "Simpler Method":
        prediction = simple_classifier(img)
        class_name = ["Malignant", "Benign", "Basal Cell"]
        string="This Image Mostly like to: "+class_name[np.argmax(prediction)]
        st.sidebar.header("""
            Model Accuracy :
        """)
        st.sidebar.write("Train Accuracy: 90%")
        st.sidebar.write("Validation Accuracy: 82%")
        st.sidebar.write("Test Accuracy: 80%")
        st.write(prediction)
        st.success(string)
    elif selective_arch == "Resnet-50":
        prediction = resnet_classifier(img)
        class_name = ["Malignant", "Benign", "Basal Cell"]
        string="This Image Mostly like to: "+class_name[np.argmax(prediction)]
        st.sidebar.header("""
            Model Accuracy :
        """)
        st.sidebar.write("Train Accuracy: 100%")
        st.sidebar.write("Validation Accuracy: 95%")
        st.sidebar.write("Test Accuracy: 95%+")
        st.write(prediction)
        st.success(string)
    else:
        st.write("Models are Not Available yet")
else:
    st.write("NILL")


# component for toggling code

st.sidebar.header("""
    Code of Intire Page :
""")
show_code = st.sidebar.checkbox("Show Code")
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    # path_file = "preprocessing.py"
    url = "https://github.com/MujtabaAhmad0928/SCRS/blob/main/"+path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if show_code:
    st.code(get_file_content_as_string("main.py"))



