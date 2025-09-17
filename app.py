import streamlit as st
import cv2
import pandas as pd
import datetime
import os

# Haar cascade file (OpenCV provides this)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# CSV for attendance
ATTENDANCE_FILE = "attendance.csv"

# Initialize attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

st.title("📸 Face Detection Attendance System (OpenCV Only)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save attendance if faces found
    if len(faces) > 0:
        name = st.text_input("Enter your name to mark attendance:")
        if st.button("Mark Attendance"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = pd.DataFrame([[name, now]], columns=["Name", "Time"])
            new_entry.to_csv(ATTENDANCE_FILE, mode="a", header=False, index=False)
            st.success(f"Attendance marked for {name} ✅")

    # Show image with detections
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

# Show attendance records
if st.checkbox("📋 Show Attendance Records"):
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)
