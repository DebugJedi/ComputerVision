from io import StringIO
from pathlib import Path
import streamlit as st
import time
# from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time
from ultralytics import YOLO

#st.set_page_config(layout = "wide")
st.set_page_config(page_title = "Yolo 11m Multiple Object Detection on Pretrained Model", page_icon="🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#################### Title #####################################################
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Yolo 11m</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Multiple Object Detection on Pretrained Model</h2>", unsafe_allow_html=True)
#st.markdown('---') # inserts underline
#st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
st.markdown('#') # inserts empty space

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b,d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in run\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key = os.path.getmtime)

#------------------Main function for Execution-----------

def main():
    model = YOLO("src/best.pt") 
    source = ("Detect From Image", "Detect From Video", "Detect From Live Feed")
    source_index = st.sidebar.selectbox("Select Activity", 
    range(len(source)), format_func = lambda x: source[x])

    custClassesLst = ["OK", "ThumbsUp", "ThumbsDown", "FingersCross", "PeaceOut", "All"]

    classes_index = st.sidebar.multiselect("Select Classes", 
                                           range(len(custClassesLst)), 
                                           format_func= lambda x: custClassesLst[x])

    isAllinList = 5 in classes_index
    if isAllinList == True:
        classes_index = classes_index.clear()
        

    print("Selected Classes: ", classes_index)

    # Parameter to setup
    deviceLst = ['cpu', '0', '1', '2', '3']

    Device = st.sidebar.selectbox("Select Devices", deviceLst, index=0)
    print("Devices: ", Device)
    MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)

    # Parameter to setup

    weights = os.path.join("src", "best.pt")

    if source_index == 0:

        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type = ['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text= "Loading....."):
                st.sidebar.text("Uploaded Pic")
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture.save(os.path.join("data", "images", uploaded_file.name))
                data_source = os.path.join('data', 'images', uploaded_file.name)
        else: #not uploaded_file
            is_valid = False
            st.sidebar.text("Please upload a image to procced")
        # else: 
        #     is_valid = False
    elif source_index == 1:
        
        uploaded_file = st.sidebar.file_uploader("Upload Video", type = ['mp4'])

        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text= 'Loading.....'):
                st.sidebar.text('Uploaded Video')
                st.sidebar.video(uploaded_file)
                with open(os.path.join('data', 'videos', uploaded_file.name), 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                data_source = os.path.join('data', 'videos', uploaded_file.name)

        else: # uploaded_file is None
            is_valid = False
            st.sidebar.text("Upload video to procced")
        # else:
        #     is_valid = False
    else:
        is_valid = False
        st.write("Live feed is currently unaviable, please choose other options...")

    #     selectedCam = st.sidebar.selectbox("select Camera",("Use Webcam", "Use Other Camera"), index=0)
    #     if selectedCam:
    #         if selectedCam == "Use Other Camera":
    #             data_source = int(1)
    #             is_valid = True

    #         else:
    #             data_source = int(0)
    #             is_valid = True
    #     else:
    #         is_valid = False
        #     st.sidebar.markdown("<strong> Press 'q' multiple times on camera window and Ctrl + C on CMD to clear camera window/exit</strong>", unsafe_allow_html=True)
    if is_valid:
        print('valid')
        # if classes_index:
        if st.button('Detect'):
            if classes_index:
                with st.spinner(text = 'Detecting, please wait.....'):
                    model.predict( 
                        source = data_source,
                        show= True, 
                        save = True, 
                        conf= MIN_SCORE_THRES
                    )
            else:
                with st.spinner(text = 'Please wait...'):
                    model.predict( 
                        source = data_source,
                        show= True, 
                        save = True, 
                        conf= MIN_SCORE_THRES
                    )
            if source_index ==0:
                with st.spinner(text= 'Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                            pathImg = os.path.join(get_detection_folder(), img)
                            st.image(pathImg)

                    st.markdown("### Output")
                    st.write("Path of Saved Images: ", pathImg)
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))
                    st.balloons()

            elif source_index ==1:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            video_file = os.path.join(get_detection_folder(), vid)
                strframe  = st.empty()
                cap = cv2.VideoCapture(video_file)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("width: ", width, "\n")
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("height: ", height, "\n")

                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        strframe.image(cv2.resize(img, (width, height)), 
                        channels='BGR', use_column_width= True)
                    else:
                        break
                
                cap.release()
                st.markdown("### Output")
                st.write("Path of saved Video: ", video_file)
                st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'label'))
                st.balloons()

            else:
                with st.spinner(text = 'Preparing stream'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            liveFeedVideoFile = os.path.join(get_detection_folder(), vid)

                    st.markdown("### Output")
                    st.write("Path of Live Feed Saved Video: ", liveFeedVideoFile)
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))
                    st.balloons()




if __name__ == '__main__':
    try:
        main()

    except SystemExit:
        pass