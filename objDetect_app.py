import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# model = YOLO("src/best.pt")

st.title("Real-Time Object Detection with YOLO")

# st.sidebar.title("Settings")
# confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.8)
# webcam = st.sidebar.checkbox("Use Webcam", value=True)
def test_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open webcam with index {index}")
        return

    print(f"Webcam {index} opened successfully.")
    cap.release()

if __name__ == "__main__":
    success = test_webcam(0)  # Try index 0 first
    if not success:
        test_webcam(1)