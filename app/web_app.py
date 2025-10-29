import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# ---------------------
# CONFIG
# ---------------------
IMG_SIZE = 224
MASK_MODEL_PATH = "models/mask_detector.h5"
AGE_MODEL_PATH = "models/age_classifier_v2.h5"
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

AGE_LABELS = ["Child", "Teen", "Adult", "Middle", "Senior"]
MASK_THRESHOLD = 0.5

# ---------------------
# LOAD MODELS
# ---------------------
@st.cache_resource
def load_models():
    mask_model = load_model(MASK_MODEL_PATH)
    age_model = load_model(AGE_MODEL_PATH)
    return mask_model, age_model

mask_model, age_model = load_models()
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

# ---------------------
# HELPERS
# ---------------------
def preprocess_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array, axis=0)

def predict_face(face_img):
    inp = preprocess_face(face_img)
    mask_pred = mask_model.predict(inp, verbose=0)[0][0]
    age_pred = age_model.predict(inp, verbose=0)[0]

    mask_label = "Unmasked Face" if mask_pred >= MASK_THRESHOLD else "Masked Face"
    mask_conf = mask_pred if mask_label == "Unmasked Face" else 1 - mask_pred

    age_idx = np.argmax(age_pred)
    age_label = AGE_LABELS[age_idx]
    age_conf = age_pred[age_idx]
    return mask_label, mask_conf, age_label, age_conf

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        mask_label, mask_conf, age_label, age_conf = predict_face(face)

        color = (0, 255, 0) if mask_label == "Mask" else (0, 0, 255)
        text1 = f"{mask_label} ({mask_conf*100:.1f}%)"
        text2 = f"{age_label} ({age_conf*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text1, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, text2, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="Face Analysis System", layout="centered")

st.title("🧠 Face Analysis System")
st.markdown("Detects **Mask Usage** + **Age Group** using deep learning.")

mode = st.radio("Choose Input Mode:", ["Upload Image", "Use Webcam"])

if mode == "Upload Image":
    file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        img = cv2.imread(tmp.name)
        result = analyze_frame(img)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")

elif mode == "Use Webcam":
    st.warning("Click 'Start' to begin camera stream and press 'Stop' to end.")
    FRAME_WINDOW = st.image([])
    run = st.checkbox("Start")

    cam = cv2.VideoCapture(0)
    while run:
        ret, frame = cam.read()
        if not ret:
            st.error("Camera not available.")
            break
        result = analyze_frame(frame)
        FRAME_WINDOW.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")

    cam.release()
    st.info("Webcam stopped.")
