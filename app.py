#import
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.title("AI Bicep Curl Counter")

run = st.checkbox("Start Camera")
FRAME = st.image([])


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


if "counter" not in st.session_state:
    st.session_state.counter = 0
if "stage" not in st.session_state:
    st.session_state.stage = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine)
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        shoulder = [lm[11].x, lm[11].y]
        elbow = [lm[13].x, lm[13].y]
        wrist = [lm[15].x, lm[15].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        
        if angle > 160:
            st.session_state.stage = "down"
        if angle < 40 and st.session_state.stage == "down":
            st.session_state.stage = "up"
            st.session_state.counter += 1

        
        cv2.putText(frame, f"Angle: {int(angle)}",
                    (30,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    cv2.putText(frame, f"Reps: {st.session_state.counter}",
                (30,80), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255,0,0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME.image(frame)

    time.sleep(0.03)

cap.release()
