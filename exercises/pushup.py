import mediapipe as mp
import numpy as np
from core.utils import calculate_angle

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark


class PushUp:
    

    def __init__(self, calib_min: float = 70.0, calib_max: float = 160.0):
        self.calib_min = calib_min
        self.calib_max = calib_max
        self._stage = "up"
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

        if angle < mid and self._stage == "up":
            self._stage = "down"

        if angle > mid and self._stage == "down":
            self._stage = "up"
            rep_done = True
            quality = self._evaluate()

        return rep_done, quality

    def get_form_score(self, angle: float) -> float:
        r = self.calib_max - self.calib_min
        if r < 1:
            return 50.0

        if self._stage == "down":
            depth_score = max(0.0, 100.0 - (angle - self.calib_min) / r * 80.0)
        else:
            depth_score = 100.0 if angle > self.calib_max * 0.9 else 70.0

        
        if len(self._angle_buffer) >= 5:
            smoothness = np.std(self._angle_buffer[-5:])
            if smoothness > 12:
                depth_score = max(0.0, depth_score - 12.0)
        return round(depth_score, 1)

    def _evaluate(self) -> str:
        if not self._angle_buffer:
            return "normal"
        min_seen = min(self._angle_buffer)
        thresh = self.calib_min + 0.25 * (self.calib_max - self.calib_min)
        if min_seen > thresh:
            return "low_angle"
        return "good"
