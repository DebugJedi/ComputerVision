import streamlit as st
import cv2
# from ultralytics import YOLO
from src import predict


def main():

    st.title("Webcam Object Detection")
    model = predict.predict_stream(True)
   
    # # Create a button to start/stop the webcam
    # if 'running' not in st.session_state:
    #     st.session_state.running = False

    # if st.button('Start' if not st.session_state.running else 'Stop'):
    #     st.session_state.running = not st.session_state.running

    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(1)

    # while st.session_state.running:
    #     _, frame = camera.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = model.predict(frame, show=False, save=False, conf=0.8)
    #     annotated_frame = results[0].plot()
    #     FRAME_WINDOW.image(annotated_frame)
    
    # camera.release()

if __name__ == "__main__":
    main()