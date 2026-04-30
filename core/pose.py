import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseDetector:
    
    

    # Earthy palette
    LANDMARK_COLOR  = (122, 140, 110)   # sage green  (BGR)
    CONNECTION_COLOR = (107, 143, 113)  # muted green (BGR)
    DOT_COLOR        = (212, 209, 204)  # off-white   (BGR)

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame):
        return self._pose.process(rgb_frame)

    def draw_landmarks(self, bgr_frame, results):
        if not results.pose_landmarks:
            return

        
        connection_spec = mp_drawing.DrawingSpec(
            color=self.CONNECTION_COLOR, thickness=2, circle_radius=1
        )
        landmark_spec = mp_drawing.DrawingSpec(
            color=self.DOT_COLOR, thickness=2, circle_radius=4
        )
        mp_drawing.draw_landmarks(
            bgr_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec,
        )

    def get_landmark(self, landmarks, index):
        lm = landmarks[index]
        return lm.x, lm.y, lm.z, lm.visibility
