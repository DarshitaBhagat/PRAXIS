import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from datetime import datetime
from core.pose import PoseDetector
from core.feedback import FeedbackGenerator
from core.utils import calculate_angle, draw_form_score_ring, draw_evaluation_bar
from exercises.bicep import BicepCurl
from exercises.squat import Squat
from exercises.pushup import PushUp
from exercises.plank import Plank

import base64

def get_img_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Page Config
st.set_page_config(
    page_title="Praxis — AI Workout Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# COLOR THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #F4F1EC !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #2C2C2A !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/*LANDING PAGE  */
.landing-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    
    background: #F4F1EC;
    padding: 6rem;
}

.brand-mark {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.35em;
    color: #7A8C6E;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(6rem, 16vw, 12rem);
    font-weight: 900;
    color: #7A8C6E;
    letter-spacing: -0.02em;
    line-height: 0.9;
    text-align: center;
    margin-bottom: 1.5rem;
}

.hero-tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: #2C2C2A;
    letter-spacing: 0.1em;
    text-align: center;
    margin-bottom: 4rem;
}

.begin-btn {
    display: inline-block;
    background: #2C2C2A;
    color: #F4F1EC !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 1.1rem 3.5rem;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.begin-btn:hover { background: #7A8C6E; }

/* ─── EXERCISE SELECTION ─── */
.select-wrap {
    min-height: 100vh;
    background: #F4F1EC;
    padding: 4rem 3rem;
}

.section-eyebrow {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    color: #7A8C6E;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #2C2C2A;
    margin-bottom: 3rem;
    line-height: 1.1;
}

.exercise-card {
    background: #EDEAE3;
    border: 1px solid #D6D3CC;
    padding: 2rem;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.exercise-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #7A8C6E;
    transform: scaleY(0);
    transition: transform 0.25s ease;
}
.exercise-card:hover::before { transform: scaleY(1); }
.exercise-card:hover { border-color: #7A8C6E; }

.ex-icon { font-size: 2rem; margin-bottom: 1rem; }
.ex-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #2C2C2A;
    margin-bottom: 0.4rem;
}
.ex-desc { font-size: 0.85rem; color: #888780; line-height: 1.5; }
.ex-target {
    margin-top: 1rem;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7A8C6E;
    font-weight: 500;
}

/* ─── CALIBRATION ─── */
.calib-wrap {
    min-height: 100vh;
    background: #2C2C2A;
    padding: 3rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.calib-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    color: #F4F1EC;
    text-align: center;
    margin-bottom: 0.5rem;
}

.calib-sub {
    font-size: 0.9rem;
    color: #7A8C6E;
    text-align: center;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}

/* ─── WORKOUT ─── */
.workout-wrap {
    background: #2C2C2A;
    min-height: 100vh;
    padding: 1.5rem;
}

.workout-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}

.workout-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #F4F1EC;
}

.stat-pill {
    background: #3C3C3A;
    border: 1px solid #4A4A48;
    border-radius: 100px;
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    color: #D6D3CC;
    font-weight: 500;
}

.feedback-bar {
    background: #3C3C3A;
    border-left: 3px solid #7A8C6E;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}

.feedback-text {
    font-size: 1rem;
    color: #F4F1EC;
    font-weight: 400;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.metric-box {
    flex: 1;
    background: #3C3C3A;
    border: 1px solid #4A4A48;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    color: #F4F1EC;
    font-weight: 700;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7A8C6E;
    margin-top: 0.25rem;
}

/* ─── SUMMARY ─── */
.summary-wrap {
    min-height: 100vh;
    background: #F4F1EC;
    padding: 5rem 3rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.summary-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    color: #2C2C2A;
    text-align: center;
    margin-bottom: 0.5rem;
}

.summary-sub {
    font-size: 0.85rem;
    color: #7A8C6E;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 3rem;
}

.summary-card {
    background: #EDEAE3;
    border: 1px solid #D6D3CC;
    padding: 2.5rem;
    width: 100%;
    max-width: 700px;
    margin-bottom: 1rem;
}

.summary-stat-label {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7A8C6E;
    margin-bottom: 0.5rem;
}

.summary-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #2C2C2A;
    font-weight: 700;
}

.insight-item {
    border-left: 2px solid #7A8C6E;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: #F4F1EC;
    font-size: 0.9rem;
    color: #2C2C2A;
}

/* ─── Streamlit Overrides ─── */
div[data-testid="stButton"] > button {
    background: #2C2C2A !important;
    color: #F4F1EC !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 2.5rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: #7A8C6E !important;
    border: none !important;
}

div[data-testid="stProgress"] > div > div {
    background: #7A8C6E !important;
}

.stSpinner > div { border-top-color: #7A8C6E !important; }

div[data-testid="stSelectbox"] > div > div {
    background: #EDEAE3 !important;
    border: 1px solid #D6D3CC !important;
    border-radius: 0 !important;
    color: #2C2C2A !important;
}

div[data-testid="stAlert"] {
    background: #EDEAE3 !important;
    border: 1px solid #D6D3CC !important;
    border-radius: 0 !important;
    color: #2C2C2A !important;
}

h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #2C2C2A !important; }

div[data-testid="stImage"] img { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Session state init
def init_state():
    defaults = {
        "page": "landing",
        "exercise": None,
        "calibrated": False,
        "running": False,
        "rep_count": 0,
        "form_score": 0.0,
        "session_scores": [],
        "session_reps": 0,
        "feedback": "Get ready...",
        "start_time": None,
        "calib_reps": 0,
        "calib_min": None,
        "calib_max": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()




def _build_eval_bar_html(angle, calib_min, calib_max):
    if calib_max > calib_min:
        progress = max(0, min(1, (angle - calib_min) / (calib_max - calib_min)))
    else:
        progress = 0.5
    pct = int(progress * 100)
    bar_color = "#7A8C6E" if pct > 60 else "#8BA8B5" if pct > 30 else "#888780"
    return f"""
    <div style="margin:0.75rem 0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
            <span style="font-size:0.65rem;letter-spacing:0.2em;color:#888780;text-transform:uppercase;">Range of Motion</span>
            <span style="font-size:0.75rem;color:#7A8C6E;">{pct}%</span>
        </div>
        <div style="background:#4A4A48;height:6px;width:100%;position:relative;">
            <div style="background:{bar_color};height:100%;width:{pct}%;transition:width 0.2s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:0.25rem;">
            <span style="font-size:0.65rem;color:#888780;">{int(calib_min)}°</span>
            <span style="font-size:0.65rem;color:#888780;">{int(calib_max)}°</span>
        </div>
    </div>
    """


#Navigation
def goto(page): st.session_state["page"] = page

EXERCISES = {
    "Bicep Curl": {
        "icon": get_img_base64("assets/bicep.png"),
        "desc": "Targets biceps brachii. Build arm strength and definition.",
        "target": "Biceps · Forearms",
        "class": BicepCurl,
    },
    "Squat": {
        "icon": get_img_base64("assets/squat.png"),
        "desc": "Full lower-body compound movement for strength and power.",
        "target": "Quads · Glutes · Hamstrings",
        "class": Squat,
    },
    "Push-Up": {
        "icon": get_img_base64("assets/pushup.png"),
        "desc": "Upper body push movement. Builds chest, shoulders, and triceps.",
        "target": "Chest · Triceps · Shoulders",
        "class": PushUp,
    },
    "Plank": {
        "icon": get_img_base64("assets/plank.png"),
        "desc": "Isometric core hold. Builds stability and postural endurance.",
        "target": "Core · Shoulders · Glutes",
        "class": Plank,
    },
}


# LANDING PAGE

if st.session_state["page"] == "landing":
    st.markdown("""
    <div class="landing-wrap">
        <div class="brand-mark">AI Workout Coach</div>
        <div class="hero-title">PRAXIS</div>
        <div class="hero-tagline">Train right. Every time.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("BEGIN WORKOUT"):
            goto("select")
            st.rerun()


#EXERCISE SELECTION

elif st.session_state["page"] == "select":
    st.markdown("""
    <div style="padding: 4rem 3rem 2rem;">
        <div class="section-eyebrow">Step 01 — Choose Exercise</div>
        <div class="section-title">What are we<br>training today?</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (name, meta) in enumerate(EXERCISES.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="exercise-card">
                <div class="ex-icon"><img src="data:image/png;base64,{meta['icon']}" width="80" /></div>
                <div class="ex-name">{name}</div>
                <div class="ex-desc">{meta['desc']}</div>
                <div class="ex-target">{meta['target']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select  →  {name}", key=f"btn_{name}"):
                st.session_state["exercise"] = name
                st.session_state["calibrated"] = False
                st.session_state["rep_count"] = 0
                st.session_state["session_scores"] = []
                st.session_state["calib_reps"] = 0
                st.session_state["calib_min"] = None
                st.session_state["calib_max"] = None
                goto("calibrate")
                st.rerun()

    st.markdown("<div style='padding: 1rem 3rem;'>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        goto("landing")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


#  CALIBRATION PAGE

elif st.session_state["page"] == "calibrate":
    ex_name = st.session_state["exercise"]
    meta = EXERCISES[ex_name]
    ExClass = meta["class"]

    st.markdown(f"""
    <div style="background:#2C2C2A; padding:3rem; min-height:100vh;">
        <div style="text-align:center; margin-bottom:2rem;">
            <div style="font-size:0.75rem;letter-spacing:0.3em;color:#7A8C6E;text-transform:uppercase;margin-bottom:0.5rem;">Calibration</div>
            <div style="font-family:'Playfair Display',serif;font-size:2.5rem;color:#F4F1EC;"><img src="data:image/png;base64,{meta['icon']}" width="80" /> {ex_name}</div>
            <div style="font-size:0.9rem;color:#888780;margin-top:0.5rem;">Perform 3 reps so Praxis can learn your range of motion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    calib_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    TARGET_CALIB_REPS = 3
    calib_done = st.session_state.get("calibrated", False)

    if not calib_done:
        detector = PoseDetector()
        ex_tracker = ExClass()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        calib_reps = 0
        all_angles = []
        frame_slot = calib_placeholder.empty()

        while calib_reps < TARGET_CALIB_REPS:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(frame_rgb)

            if results.pose_landmarks:
                detector.draw_landmarks(frame, results)
                angle = ex_tracker.get_primary_angle(results.pose_landmarks.landmark)
                if angle is not None:
                    all_angles.append(angle)
                    rep_done, _ = ex_tracker.update(angle, calibrating=True)
                    if rep_done:
                        calib_reps += 1

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 70), (44, 44, 42), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, f"CALIBRATING  {calib_reps}/{TARGET_CALIB_REPS} REPS",
                        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (122, 140, 110), 2)

            frame_slot.image(frame, channels="BGR", use_container_width=True)

            prog = calib_reps / TARGET_CALIB_REPS
            progress_placeholder.progress(prog)
            status_placeholder.markdown(
                f"<div style='text-align:center;color:#7A8C6E;font-size:0.85rem;letter-spacing:0.1em;'>"
                f"Rep {calib_reps} of {TARGET_CALIB_REPS} complete</div>",
                unsafe_allow_html=True
            )

        cap.release()

        if all_angles:
            st.session_state["calib_min"] = float(np.percentile(all_angles, 5))
            st.session_state["calib_max"] = float(np.percentile(all_angles, 95))
            st.session_state["calibrated"] = True

        calib_placeholder.empty()
        progress_placeholder.empty()
        status_placeholder.empty()
        st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">✓</div>
            <div style="font-family:'Playfair Display',serif;font-size:2rem;color:#F4F1EC;margin-bottom:0.5rem;">Calibration Complete</div>
            <div style="color:#7A8C6E;font-size:0.9rem;margin-bottom:2rem;">Your motion range has been recorded. Ready to begin.</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("START WORKOUT →"):
                st.session_state["rep_count"] = 0
                st.session_state["session_scores"] = []
                st.session_state["start_time"] = time.time()
                goto("workout")
                st.rerun()


# WORKOUT MAIN

elif st.session_state["page"] == "workout":
    ex_name = st.session_state["exercise"]
    meta = EXERCISES[ex_name]
    ExClass = meta["class"]

    calib_min = st.session_state.get("calib_min", 60.0)
    calib_max = st.session_state.get("calib_max", 160.0)

    st.markdown(f"""
    <div style="background:#2C2C2A;padding:1.5rem 2rem 0.5rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-size:0.7rem;letter-spacing:0.25em;color:#7A8C6E;text-transform:uppercase;">Now Training</div>
                <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#F4F1EC;display:flex;align-items:center;gap:10px;">
                    <img src="data:image/png;base64,{meta['icon']}" style="width:28px;height:28px;" />
                    {ex_name}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.7rem;letter-spacing:0.2em;color:#888780;text-transform:uppercase;">Target Muscles</div>
                <div style="font-size:0.85rem;color:#7A8C6E;">{meta['target']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cam, col_stats = st.columns([3, 2])

    with col_cam:
        frame_slot = st.empty()

    with col_stats:
        reps_slot = st.empty()
        score_slot = st.empty()
        eval_slot = st.empty()
        feedback_slot = st.empty()

    complete_col1, complete_col2, complete_col3 = st.columns([1, 2, 1])
    with complete_col2:
        done_btn = st.button("✓  EXERCISE COMPLETED")

    if done_btn:
        goto("summary")
        st.rerun()

    detector = PoseDetector()
    ex_tracker = ExClass(calib_min=calib_min, calib_max=calib_max)
    feedback_gen = FeedbackGenerator()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    rep_count = st.session_state.get("rep_count", 0)
    session_scores = st.session_state.get("session_scores", [])
    current_score = 0.0
    current_feedback = "Begin your reps..."
    current_angle = 0.0

    for _ in range(5000):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        if results.pose_landmarks:
            detector.draw_landmarks(frame, results)
            angle = ex_tracker.get_primary_angle(results.pose_landmarks.landmark)

            if angle is not None:
                current_angle = angle
                rep_done, quality = ex_tracker.update(angle)
                score = ex_tracker.get_form_score(angle)
                current_score = score

                if rep_done:
                    rep_count += 1
                    session_scores.append(score)
                    st.session_state["rep_count"] = rep_count
                    st.session_state["session_scores"] = session_scores

                current_feedback = feedback_gen.get_feedback(ex_name, angle, score, quality)

                # Draw score ring on frame
                frame = draw_form_score_ring(frame, score)

                # Draw angle arc
                cv2.putText(frame, f"{int(angle)}", (560, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (214, 211, 204), 1)

        frame_slot.image(frame, channels="BGR", use_container_width=True)

        avg_score = float(np.mean(session_scores)) if session_scores else current_score
        score_color = "#6B8F71" if current_score >= 75 else "#8BA8B5" if current_score >= 50 else "#D6D3CC"

        reps_slot.markdown(f"""
        <div style="display:flex;gap:1rem;margin:0.75rem 0;">
            <div style="flex:1;background:#3C3C3A;border:1px solid #4A4A48;padding:1.25rem;text-align:center;">
                <div style="font-family:'Playfair Display',serif;font-size:3rem;color:#F4F1EC;font-weight:700;line-height:1;">{rep_count}</div>
                <div style="font-size:0.65rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-top:0.4rem;">Reps</div>
            </div>
            <div style="flex:1;background:#3C3C3A;border:1px solid #4A4A48;padding:1.25rem;text-align:center;">
                <div style="font-family:'Playfair Display',serif;font-size:3rem;color:{score_color};font-weight:700;line-height:1;">{int(current_score)}</div>
                <div style="font-size:0.65rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-top:0.4rem;">Form Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        eval_bar_html = _build_eval_bar_html(current_angle, calib_min, calib_max)
        eval_slot.markdown(eval_bar_html, unsafe_allow_html=True)

        fb_color = "#7A8C6E" if current_score >= 75 else "#8BA8B5" if current_score >= 50 else "#888780"
        feedback_slot.markdown(f"""
        <div style="background:#3C3C3A;border-left:3px solid {fb_color};padding:1rem 1.25rem;margin-top:0.5rem;">
            <div style="font-size:0.65rem;letter-spacing:0.2em;color:#888780;text-transform:uppercase;margin-bottom:0.4rem;">Coach Feedback</div>
            <div style="font-size:1rem;color:#F4F1EC;">{current_feedback}</div>
        </div>
        <div style="background:#3C3C3A;border:1px solid #4A4A48;padding:0.75rem 1.25rem;margin-top:0.75rem;">
            <div style="font-size:0.65rem;letter-spacing:0.2em;color:#888780;text-transform:uppercase;margin-bottom:0.3rem;">Session Avg Score</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.5rem;color:#F4F1EC;">{int(avg_score)}<span style="font-size:0.8rem;color:#888780;">/100</span></div>
        </div>
        """, unsafe_allow_html=True)

    cap.release()



# SESSION SUMMARY

elif st.session_state["page"] == "summary":
    rep_count = st.session_state.get("rep_count", 0)
    session_scores = st.session_state.get("session_scores", [])
    ex_name = st.session_state.get("exercise", "Workout")
    start_time = st.session_state.get("start_time", None)

    avg_score = float(np.mean(session_scores)) if session_scores else 0.0
    duration_s = int(time.time() - start_time) if start_time else 0
    duration_str = f"{duration_s // 60}m {duration_s % 60}s"

    grade = "Excellent" if avg_score >= 85 else "Good" if avg_score >= 70 else "Developing" if avg_score >= 50 else "Needs Work"
    insights = []
    if avg_score >= 80:
        insights.append("Outstanding form consistency — keep maintaining this standard.")
    elif avg_score >= 65:
        insights.append("Good range of motion. Focus on the bottom phase for deeper activation.")
    else:
        insights.append("Prioritize full range of motion over speed.")

    if rep_count >= 15:
        insights.append(f"Strong volume — {rep_count} reps completed. Consider progressive overload.")
    elif rep_count >= 8:
        insights.append(f"{rep_count} reps completed. Aim for 12–15 next session.")
    else:
        insights.append(f"{rep_count} reps completed. Build endurance gradually.")

    insights.append("Calibration data saved. Next session will be more accurate.")

    insights_html = "".join(
        [
            f"""
            <div style="border-left:2px solid #7A8C6E; padding:0.6rem 1rem; margin-bottom:0.6rem; background:#F4F1EC; font-size:0.9rem; color:#2C2C2A;">{i}</div>
            """
            for i in insights
        ]
    )

    summary_html = f"""
    <div style="background:#F4F1EC;padding:3rem 3rem 1rem 3rem;display:flex;flex-direction:column;align-items:center;">
        <div style="font-size:0.75rem;letter-spacing:0.3em;color:#7A8C6E;text-transform:uppercase;margin-bottom:0.75rem;text-align:center;">Session Complete</div>
        <div style="font-family:'Playfair Display',serif;font-size:4rem;color:#2C2C2A;text-align:center;margin-bottom:0.25rem;line-height:1;">Well done.</div>
        <div style="font-size:0.85rem;color:#888780;letter-spacing:0.1em;margin-bottom:3rem;text-align:center;">{ex_name} · {duration_str}</div>

        <div style="display:flex;gap:1rem;width:100%;max-width:700px;margin-bottom:1rem;">
            <div style="flex:1;background:#EDEAE3;border:1px solid #D6D3CC;padding:2rem;text-align:center;">
                <div style="font-size:0.65rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-bottom:0.5rem;">Total Reps</div>
                <div style="font-family:'Playfair Display',serif;font-size:4rem;color:#2C2C2A;font-weight:700;line-height:1;">{rep_count}</div>
            </div>
            <div style="flex:1;background:#EDEAE3;border:1px solid #D6D3CC;padding:2rem;text-align:center;">
                <div style="font-size:0.65rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-bottom:0.5rem;">Avg Form Score</div>
                <div style="font-family:'Playfair Display',serif;font-size:4rem;color:#2C2C2A;font-weight:700;line-height:1;">{int(avg_score)}</div>
            </div>
            <div style="flex:1;background:#2C2C2A;padding:2rem;text-align:center;">
                <div style="font-size:0.65rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-bottom:0.5rem;">Grade</div>
                <div style="font-family:'Playfair Display',serif;font-size:2rem;color:#F4F1EC;font-weight:700;line-height:1.2;">{grade}</div>
            </div>
        </div>

        <div style="width:100%;max-width:700px;background:#EDEAE3;border:1px solid #D6D3CC;padding:2rem;margin-bottom:2rem;">
            <div style="font-size:0.7rem;letter-spacing:0.2em;color:#7A8C6E;text-transform:uppercase;margin-bottom:1rem;">Performance Insights</div>
            {insights_html}
        </div>
    </div>
    """
    import streamlit.components.v1 as components
    components.html(summary_html, height=700, scrolling=True)

    col_outer1, col_center, col_outer2 = st.columns([1, 2, 1])

    with col_center:
        col1, col2, col3 = st.columns(3 , gap="large")
    
    
        with col1:
            if st.button("← New Exercise"):
                st.session_state["calibrated"] = False
                goto("select")
                st.rerun()
        with col2:
            if st.button("↺ Same Exercise"):
                st.session_state["rep_count"] = 0
                st.session_state["session_scores"] = []
                st.session_state["start_time"] = time.time()
                goto("workout")
                st.rerun()
        with col3:
            if st.button("⌂ Home Page "):
                for k in ["exercise", "calibrated", "rep_count", "session_scores", "start_time",
                        "calib_min", "calib_max", "calib_reps"]:
                    st.session_state[k] = None if k != "rep_count" else 0
                goto("landing")
                st.rerun()
