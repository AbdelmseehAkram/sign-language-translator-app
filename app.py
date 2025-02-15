import av
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import gdown
import os

# Model configuration
MODEL_URL = "https://drive.google.com/uc?id=1mnasQGJhxbxGW1wotIT1nr1icNzz9xdC"
MODEL_PATH = "model_alphabet_transfer.keras"

# Model loading with caching
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

# Load model
model = load_my_model()
class_labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# App interface
st.title("👋 Sign Language Translator")
st.markdown("Press the button below to start real-time detection")

# Video processing class
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.model = model
        self.class_labels = class_labels

    def _preprocess(self, frame):
        """Process frame for model input"""
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def transform(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            print("Shape before processing:", img.shape)  # Debugging

            # Make prediction
            input_tensor = self._preprocess(img)
            print("Shape after processing:", input_tensor.shape)  # Debugging

            preds = self.model.predict(input_tensor)
            class_idx = np.argmax(preds)
            confidence = np.max(preds)

            # Add prediction overlay
            label = f"{self.class_labels[class_idx]} ({confidence:.2f})"
            cv2.putText(
                img, 
                label,
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2,
                cv2.LINE_AA
            )
            
            return img
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return frame.to_ndarray(format="bgr24")

# WebRTC streamer configuration
ctx = webrtc_streamer(
    key="sign-language-detector",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignLanguageTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,  # Disabled async processing to avoid issues
)