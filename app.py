import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import datetime

# Paths
DATASET_PATH = "images"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Prepare attendance file
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Train recognizer from dataset
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map, id_counter = {}, 0

    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        detected = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(detected) == 0:
            continue
        (x, y, w, h) = detected[0]
        faces.append(gray[y:y+h, x:x+w])
        name = filename.split(".")[0]  # "alice.jpg" → "alice"
        if name not in label_map:
            label_map[name] = id_counter
            id_counter += 1
        labels.append(label_map[name])

    if faces:
        recognizer.train(faces, np.array(labels))
    return recognizer, {v: k for k, v in label_map.items()}

# Train model once
recognizer, id_to_name = train_recognizer()

st.title("🎓 Face Recognition Attendance System")

# Upload image
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(roi_gray)
        name = id_to_name.get(label_id, "Unknown")

        # Draw results
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{name} ({int(confidence)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mark attendance if recognized
        if name != "Unknown":
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = pd.DataFrame([[name, now]], columns=["Name", "Time"])
            new_entry.to_csv(ATTENDANCE_FILE, mode="a", header=False, index=False)
            st.success(f"Attendance marked for {name} ✅")

    # Show image with results
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

# Show attendance log
if st.checkbox("📋 Show Attendance Records"):
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)
