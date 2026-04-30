import mediapipe as mp
import numpy as np
import time
from core.utils import calculate_angle

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark


class Plank:
    

    def __init__(self, calib_min: float = 155.0, calib_max: float = 180.0):
        self.calib_min = calib_min
        self.calib_max = calib_max
        self._hold_start = None
        self._last_rep_time = None
        self._angle_buffer = []
        self._REP_INTERVAL = 5.0   # count 1 "rep" per 5 seconds of hold

    def get_primary_angle(self, landmarks) -> float | None:
        try:
            r_vis = landmarks[L.RIGHT_ANKLE.value].visibility
            l_vis = landmarks[L.LEFT_ANKLE.value].visibility

            if r_vis >= l_vis:
                a = [landmarks[L.RIGHT_SHOULDER.value].x, landmarks[L.RIGHT_SHOULDER.value].y]
                b = [landmarks[L.RIGHT_HIP.value].x,      landmarks[L.RIGHT_HIP.value].y]
                c = [landmarks[L.RIGHT_ANKLE.value].x,    landmarks[L.RIGHT_ANKLE.value].y]
            else:
                a = [landmarks[L.LEFT_SHOULDER.value].x,  landmarks[L.LEFT_SHOULDER.value].y]
                b = [landmarks[L.LEFT_HIP.value].x,       landmarks[L.LEFT_HIP.value].y]
                c = [landmarks[L.LEFT_ANKLE.value].x,     landmarks[L.LEFT_ANKLE.value].y]

            return calculate_angle(a, b, c)
        except Exception:
            return None

    def update(self, angle: float, calibrating: bool = False):
        self._angle_buffer.append(angle)
        if len(self._angle_buffer) > 15:
            self._angle_buffer.pop(0)

        in_position = self.calib_min - 15 <= angle <= 180

        rep_done = False
        quality = "normal"
        now = time.time()

        if in_position:
            if self._hold_start is None:
                self._hold_start = now
            held = now - self._hold_start
            if self._last_rep_time is None:
                self._last_rep_time = now
            if now - self._last_rep_time >= self._REP_INTERVAL:
                rep_done = True
                self._last_rep_time = now
                quality = self._evaluate(angle)
        else:
            self._hold_start = None
            if angle < self.calib_min - 15:
                quality = "high_angle"   # hips too high
            elif angle > 180:
                quality = "low_angle"    # hips sagging

        return rep_done, quality

    def get_form_score(self, angle: float) -> float:
        target = (self.calib_min + 180) / 2
        deviation = abs(angle - target)
        score = max(0.0, 100.0 - deviation * 2.0)

        if len(self._angle_buffer) >= 8:
            stability = np.std(self._angle_buffer[-8:])
            if stability < 3:
                score = min(100.0, score + 10.0)
            elif stability > 10:
                score = max(0.0, score - 20.0)
        return round(score, 1)

    def _evaluate(self, angle: float) -> str:
        if not self._angle_buffer:
            return "normal"
        stability = np.std(self._angle_buffer[-8:]) if len(self._angle_buffer) >= 8 else 99
        if stability > 8:
            return "unstable"
        return "good"
