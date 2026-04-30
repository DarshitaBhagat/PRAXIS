import mediapipe as mp
import numpy as np
from core.utils import calculate_angle

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark


class BicepCurl:
    
    def __init__(self, calib_min: float = 40.0, calib_max: float = 160.0):
        self.calib_min = calib_min
        self.calib_max = calib_max
        self._stage = "down"  
        self._prev_angle = None
        self._angle_buffer = []

    

    def get_primary_angle(self, landmarks) -> float | None:
        try:
            
            r_vis = landmarks[L.RIGHT_WRIST.value].visibility
            l_vis = landmarks[L.LEFT_WRIST.value].visibility

            if r_vis >= l_vis:
                a = [landmarks[L.RIGHT_SHOULDER.value].x, landmarks[L.RIGHT_SHOULDER.value].y]
                b = [landmarks[L.RIGHT_ELBOW.value].x,    landmarks[L.RIGHT_ELBOW.value].y]
                c = [landmarks[L.RIGHT_WRIST.value].x,    landmarks[L.RIGHT_WRIST.value].y]
            else:
                a = [landmarks[L.LEFT_SHOULDER.value].x,  landmarks[L.LEFT_SHOULDER.value].y]
                b = [landmarks[L.LEFT_ELBOW.value].x,     landmarks[L.LEFT_ELBOW.value].y]
                c = [landmarks[L.LEFT_WRIST.value].x,     landmarks[L.LEFT_WRIST.value].y]

            return calculate_angle(a, b, c)
        except Exception:
            return None

    def update(self, angle: float, calibrating: bool = False):
        
        self._angle_buffer.append(angle)
        if len(self._angle_buffer) > 10:
            self._angle_buffer.pop(0)

        rep_done = False
        quality = "normal"

        mid = (self.calib_min + self.calib_max) / 2

        if angle < mid and self._stage == "down":
            self._stage = "up"

        if angle > mid and self._stage == "up":
            self._stage = "down"
            rep_done = True
            quality = self._evaluate()

        self._prev_angle = angle
        return rep_done, quality

    def get_form_score(self, angle: float) -> float:
        
        r = self.calib_max - self.calib_min
        if r < 1:
            return 50.0
        mid = (self.calib_min + self.calib_max) / 2
        deviation = abs(angle - mid) / (r / 2)
        score = max(0.0, 100.0 - deviation * 30.0)

        
        if len(self._angle_buffer) >= 5:
            smoothness = np.std(self._angle_buffer[-5:])
            if smoothness < 5:
                score = min(100.0, score + 10.0)
            elif smoothness > 20:
                score = max(0.0, score - 15.0)
        return round(score, 1)

    

    def _evaluate(self) -> str:
        if not self._angle_buffer:
            return "normal"
        min_seen = min(self._angle_buffer)
        max_seen = max(self._angle_buffer)

        thresh_low  = self.calib_min + 0.2 * (self.calib_max - self.calib_min)
        thresh_high = self.calib_max - 0.2 * (self.calib_max - self.calib_min)

        if min_seen > thresh_low:
            return "low_angle"
        if max_seen < thresh_high:
            return "high_angle"
        return "good"
