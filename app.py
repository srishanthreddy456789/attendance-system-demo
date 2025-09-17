import streamlit as st
import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Attendance System", layout="wide")
st.title("🎓 Automated Attendance System (Face Recognition Demo)")

# ------------------ Load known faces ------------------
path = "known_faces"
images = []
student_names = []
if os.path.exists(path):
    for filename in os.listdir(path):
        img = face_recognition.load_image_file(f"{path}/{filename}")
        images.append(img)
        student_names.append(os.path.splitext(filename)[0])
else:
    os.makedirs(path)

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # avoid error if no face in image
            encode_list.append(encodes[0])
    return encode_list

encode_list_known = find_encodings(images)
st.success(f"Loaded {len(encode_list_known)} student(s) from known_faces/")

# ------------------ Attendance ------------------
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    if name not in df["Name"].values:
        now = datetime.now()
        df = pd.concat([df, pd.DataFrame([[name, now.strftime("%H:%M:%S")]], columns=["Name", "Time"])], ignore_index=True)
        df.to_csv(attendance_file, index=False)

# ------------------ Streamlit Webcam ------------------
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Webcam not detected.")
        break

    img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(rgb_small)
    encodes_cur_frame = face_recognition.face_encodings(rgb_small, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)

        if len(face_dis) > 0:
            match_index = face_dis.argmin()
            if matches[match_index]:
                name = student_names[match_index].upper()
                y1, x2, y2, x1
