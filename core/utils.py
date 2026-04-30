import cv2
import numpy as np




def calculate_angle(a, b, c):
    
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# Form score ring 

def draw_form_score_ring(frame, score: float) -> np.ndarray:
    """
    Draws a circular form-score indicator in the top-right corner.
    score: 0–100.
    """
    h, w = frame.shape[:2]
    cx, cy, radius = w - 60, 60, 45
    thickness = 6

    
    cv2.circle(frame, (cx, cy), radius, (60, 60, 58), thickness)

    # Score arc
    angle = int(score / 100 * 360)
    if score >= 75:
        color = (110, 143, 107)   # sage  (BGR)
    elif score >= 50:
        color = (181, 168, 139)   # soft blue-ish (BGR)
    else:
        color = (140, 140, 136)   # grey

    cv2.ellipse(
        frame, (cx, cy), (radius, radius),
        -90, 0, angle, color, thickness, lineType=cv2.LINE_AA
    )

    
    label = str(int(score))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.6, 2)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(frame, label, (tx, ty), font, 0.6, (212, 209, 204), 2, cv2.LINE_AA)

    
    cv2.putText(frame, "FORM", (cx - 18, cy + radius + 18),
                font, 0.35, (122, 140, 110), 1, cv2.LINE_AA)

    return frame


# Evaluation Bar

def draw_evaluation_bar(frame, angle: float, calib_min: float, calib_max: float) -> np.ndarray:
    """
    Draws a horizontal range-of-motion bar at the bottom of the frame.
    """
    h, w = frame.shape[:2]
    bar_x, bar_y = 20, h - 30
    bar_w = w - 40
    bar_h = 8

    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 58), -1)

    
    if calib_max > calib_min:
        pct = max(0.0, min(1.0, (angle - calib_min) / (calib_max - calib_min)))
    else:
        pct = 0.5

    fill = int(bar_w * pct)
    bar_color = (110, 143, 107) if pct > 0.6 else (139, 168, 181) if pct > 0.3 else (136, 135, 128)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{int(calib_min)}", (bar_x, bar_y - 6),
                font, 0.38, (136, 135, 128), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(calib_max)}", (bar_x + bar_w - 30, bar_y - 6),
                font, 0.38, (136, 135, 128), 1, cv2.LINE_AA)
    cv2.putText(frame, "RANGE", (bar_x + bar_w // 2 - 20, bar_y - 6),
                font, 0.38, (122, 140, 110), 1, cv2.LINE_AA)

    return frame
