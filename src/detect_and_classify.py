import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 224
MASK_MODEL_PATH = "models/mask_detector.h5"
AGE_MODEL_PATH = "models/age_classifier_v2.h5"
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

AGE_LABELS = ["Child", "Teen", "Adult", "Middle", "Senior"]
MASK_THRESHOLD = 0.5  # sigmoid threshold for mask model

# -------------------------
# LOAD MODELS
# -------------------------
print("[INFO] Loading models...")
mask_model = load_model(MASK_MODEL_PATH)
age_model = load_model(AGE_MODEL_PATH)

# -------------------------
# LOAD FACE DETECTOR
# -------------------------
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
if face_cascade.empty():
    raise IOError("Could not load Haar cascade XML. Check your OpenCV installation.")

# -------------------------
# PREPROCESS HELPERS
# -------------------------
def preprocess_face_for_model(face_img):
    """
    face_img: BGR image crop (numpy array)
    returns: (1, IMG_SIZE, IMG_SIZE, 3) normalized float32
    """
    # convert BGR -> RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array.astype('float32'), axis=0)

# -------------------------
# RUN REAL-TIME INFERENCE
# -------------------------
def run_webcam(source=0, save_output=False, output_path="output.avi"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {source}")

    # video writer if requested
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optional: resize to speed up detection (keeps aspect ratio)
        scale_percent = 100  # change to 75 or 50 if you need more speed
        # frame = cv2.resize(frame, (0,0), fx=scale_percent/100, fy=scale_percent/100)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces - returns list of (x, y, w, h)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # expand bounding box slightly for safer crop
            pad = int(0.1 * w)
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # preprocess for both models
            inp = preprocess_face_for_model(face_crop)

            # mask prediction (sigmoid output -> float between 0 and 1)
            mask_pred = mask_model.predict(inp)[0][0]
            mask_label = "Unmasked Face" if mask_pred >= MASK_THRESHOLD else "Masked Face"
            mask_conf = mask_pred if mask_pred >= MASK_THRESHOLD else 1 - mask_pred

            # age prediction (softmax -> argmax)
            age_pred = age_model.predict(inp)[0]
            age_idx = np.argmax(age_pred)
            age_label = AGE_LABELS[age_idx]
            age_conf = age_pred[age_idx]

            # prepare display text
            text1 = f"{mask_label}: {mask_conf*100:.1f}%"
            text2 = f"{age_label}: {age_conf*100:.1f}%"

            # choose box color
            if mask_label == "Mask":
                box_color = (0, 255, 0)  # green
            else:
                box_color = (0, 0, 255)  # red

            # draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, text1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(frame, text2, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Analysis — Mask + Age", frame)

        if save_output and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # To run webcam: python src/detect_and_classify.py
    # To run with video file: set source to a path string e.g. run_webcam("videos/test.mp4")
    run_webcam(source=0, save_output=False)
