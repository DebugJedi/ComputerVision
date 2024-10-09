import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, ClientSettings
from ultralytics import YOLO
import cv2

# Define the transformer class
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO("src/best.pt")  # Ensure the path is correct
        self.conf_threshold = 0.8

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(source=img, conf=self.conf_threshold, save=False)
        annotated_img = results[0].plot()
        return annotated_img

def main():
    st.title("Real-Time Object Detection with YOLO and Streamlit")

    # Sidebar settings
    st.sidebar.title("Settings")
    model_path = st.sidebar.text_input("YOLO Model Path", "src/best.pt")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.8)

    # Update the transformer with new settings if changed
    class YOLOTransformerCustom(VideoTransformerBase):
        def __init__(self):
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(source=img, conf=self.conf_threshold, save=False)
            annotated_img = results[0].plot()
            return annotated_img

    # WebRTC Configuration
    WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    # Initialize the webrtc streamer
    webrtc_streamer(
        key="yolo-object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=YOLOTransformerCustom,
        async_transform=True,
    )

if __name__ == "__main__":
    main()
