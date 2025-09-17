import streamlit as st
import cv2
from datetime import datetime
import pandas as pd
import os

st.title("Automated Attendance System (Demo)")
st.write("This demo uses webcam + face detection to mark attendance.")

# Initialize attendance file
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv("attendance.csv", index=False)

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv("attendance.csv")
    if name not in df["Name"].values:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        df = pd.concat([df, pd.DataFrame([[name, time_string]], columns=["Name", "Time"])], ignore_index=True)
        df.to_csv("attendance.csv", index=False)

# Start webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to access webcam")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        mark_attendance("Student")  # Dummy name for demo
    
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()

st.subheader("📋 Attendance Log")
st.dataframe(pd.read_csv("attendance.csv"))

