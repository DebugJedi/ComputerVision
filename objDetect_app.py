import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("src/best.pt")

model = load_model()

st.title("Real-Time Object Detection with YOLO")

# Add a button to start/stop the webcam
run = st.button("Start/Stop Webcam")

# Placeholder for video frames
stframe = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame from webcam")
        break

    # Perform object detection
    results = model.predict(source=frame, conf=0.5, save=False)

    # Annotate frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display in Streamlit
    stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    # Check if the stop button is pressed
    if not run:
        break

# Release the webcam when stopped
cap.release()