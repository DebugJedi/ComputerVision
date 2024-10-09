import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np
import time
model = YOLO("src/best.pt")

st.title("Real-Time Object Detection with YOLO")
# Initialize webcam
cap = cv2.VideoCapture(0)

# Placeholder for video frames
stframe = st.empty()

# Stream video frames
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        break

    # Perform object detection
    results = model.predict(source=frame, conf=0.8, save=False)

    # Annotate frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display in Streamlit
    stframe.image(annotated_frame, channels="RGB")

    # Control frame rate
    time.sleep(0.03)  # Approximately 30 FPS
