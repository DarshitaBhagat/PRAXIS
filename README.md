##🏋️ Praxis — AI-Powered Workout Coach

Praxis is a real-time computer vision system that detects exercises, tracks repetitions, evaluates form, and provides instant feedback — turning a simple workout tracker into an intelligent fitness assistant.

---

##  What It Does

Praxis uses your webcam to:

* Detect body posture using pose estimation
* Track exercises:

  * Bicep curls
  * Squats
  * Push-ups
  * Plank
* Count repetitions automatically
* Analyze movement quality using joint angles
* Provide **real-time feedback**
* Score form dynamically based on **your own motion range**

---

## Core Idea

> Praxis adapts to your body — learning your movement range and evaluating performance relative to your own motion patterns instead of fixed thresholds.

---

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* Streamlit

---

##  System Architecture

```id="arch-praxis"
[ Webcam Input ]
        ↓
[ Frame Capture (OpenCV) ]
        ↓
[ Pose Detection (MediaPipe) ]
        ↓
[ Feature Extraction (Joint Angles) ]
        ↓
[ Motion Range Tracker (Adaptive Calibration) ]
        ↓
[ Rep Counter + Form Analysis ]
        ↓
[ Feedback Generator ]
        ↓
[ Streamlit UI ]
```

---

##  How It Works

1. Capture webcam frames
2. Detect body landmarks
3. Compute joint angles
4. Learn **min–max motion range** via calibration
5. Detect reps using relative thresholds
6. Evaluate:

   * Range of motion
   * Stability
   * Alignment
7. Generate feedback
8. Display results in real time

---

#  Features

---

##  Real-Time Pose Detection

Extracts 33 body keypoints using MediaPipe.

---

## Adaptive Rep Counting

Counts reps based on **relative motion range**, not fixed thresholds.

---

## Form Analysis Engine

Evaluates:

* Range of motion
* Stability (smoothness)
* Alignment (posture correctness)

---

## Instant Feedback

Examples:

* “Go lower”
* “Keep your back straight”
* “Control your movement”

---

## Plank Detection

* Tracks hold time
* Measures posture stability
* Detects deviation from baseline

---

##  Form Score Ring

A circular progress indicator showing real-time form score.

* Smooth animated updates
* Color-coded feedback:

  * 🟢 Good
  * 🟡 Needs improvement
  * 🔴 Poor

 Makes feedback instantly understandable

---

##  Smooth Transitions 

* Rep counter animations
* Score updates with easing
* UI transitions for better flow

 Makes the app feel responsive and premium

---

## Focus Mode UI 

Minimal interface during workouts:

* Webcam feed
* Form score
* Reps
* Feedback

Removes distractions and improves clarity

---

## Calibration UX 

Before starting:

* User performs 2–3 reps
* System learns:

  * min angle
  * max angle

Includes:

* progress indicator
* “Calibrating…” state

Ensures accuracy across users

---

## Session Summary 

After workout:

* Total reps
* Average form score
* Performance insights

Turns tracking into progress awareness

---

# What Makes Praxis Different

* Not just a rep counter
* Uses **adaptive motion modeling**
* Provides **real-time actionable feedback**
* Clean, focused UX design
* Built as a modular and extensible system

---

## Project Structure

```id="struct-praxis"
praxis/
│
├── app.py
│
├── core/
│   ├── pose.py
│   ├── utils.py
│   ├── feedback.py
│
├── exercises/
│   ├── bicep.py
│   ├── squat.py
│   ├── pushup.py
│   ├── plank.py
│
├── assets/
│   ├── bicep.png
│   ├── plank.png
│   ├── pushup.png
│   ├──squat.png
│
└── requirements.txt
```

---

## Getting Started

### 1. Clone repo

```bash id="clone-praxis"
git clone https://github.com/yourusername/praxis.git
cd praxis
```

### 2. Install dependencies

```bash id="install-praxis"
pip install -r requirements.txt
```

### 3. Run app

```bash id="run-praxis"
streamlit run app.py
```

---

##  Usage

1. Select exercise
2. Complete calibration
3. Start workout
4. Get:

   * Rep count
   * Form score (live ring)
   * Real-time feedback
5. View session summary

---

##  Notes

* Ensure good lighting
* Keep full body in frame
* Stable camera improves accuracy
* Calibration is important for best results

---

##  Future Improvements

* Voice feedback (AuraOS integration)
* ML-based form correction
* Personalized workout plans
* Mobile/web deployment
* Multi-user tracking

---

