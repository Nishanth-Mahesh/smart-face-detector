import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Smart Face Detection", layout="centered")
st.title("🧠 Smart Face Detection App (Offline)")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Face Detection Function
def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_np, len(faces)

# User input mode
mode = st.radio("Choose Input Method", ["📸 Webcam", "📁 Upload Image"])

if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        result, count = detect_faces(image_np)
        st.image(result, channels="BGR", caption=f"✅ Detected {count} face(s)")

elif mode == "📸 Webcam":
    st.warning("Click 'Start' to activate your webcam")
    camera = st.camera_input("Take a picture")

    if camera is not None:
        image = Image.open(camera)
        image = image.convert("RGB")
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        result, count = detect_faces(image_np)
        st.image(result, channels="BGR", caption=f"✅ Detected {count} face(s)")
