import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

model = YOLO("src/best.pt")

st.title("Real-Time Object Detection with YOLO")

# st.sidebar.title("Settings")
# confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.8)
# webcam = st.sidebar.checkbox("Use Webcam", value=True)
