import random


class FeedbackGenerator:
    

    _GENERIC_GOOD = [
        "Great form — keep it up!",
        "Excellent control. Stay consistent.",
        "Perfect execution. Well done.",
        "Strong rep. Maintain this quality.",
    ]

    _GENERIC_BAD = [
        "Control your movement.",
        "Slow down — quality over speed.",
        "Focus on full range of motion.",
    ]

    _RULES = {
        "Bicep Curl": {
            "low_angle": "Curl higher — squeeze the bicep at the top.",
            "high_angle": "Extend fully at the bottom for full range.",
            "unstable": "Keep your elbow fixed. Don't swing the arm.",
            "good": ["Tight curl — perfect contraction.", "Great bicep engagement.", "Full range achieved."],
        },
        "Squat": {
            "low_angle": "Drive those hips back and down lower.",
            "high_angle": "Stand tall at the top — lock out the rep.",
            "unstable": "Keep your chest up and knees tracking over toes.",
            "good": ["Deep squat — excellent depth.", "Strong drive through the heels.", "Great squat mechanics."],
        },
        "Push-Up": {
            "low_angle": "Lower your chest closer to the floor.",
            "high_angle": "Fully extend your elbows at the top.",
            "unstable": "Keep your core tight. Don't let your hips sag.",
            "good": ["Solid push-up. Full extension.", "Great chest activation.", "Strong lockout at the top."],
        },
        "Plank": {
            "low_angle": "Raise your hips slightly — keep a straight line.",
            "high_angle": "Lower your hips — avoid piking up.",
            "unstable": "Breathe steadily. Don't hold your breath.",
            "good": ["Rock solid plank. Keep holding.", "Perfect alignment. Stay strong.", "Great stability."],
        },
    }

    def get_feedback(self, exercise: str, angle: float, score: float, quality: str = "normal") -> str:
        rules = self._RULES.get(exercise, {})

        if score >= 80:
            good_msgs = rules.get("good", self._GENERIC_GOOD)
            return random.choice(good_msgs) if random.random() < 0.4 else good_msgs[0]

        if quality == "low_angle":
            return rules.get("low_angle", "Go deeper into the movement.")
        if quality == "high_angle":
            return rules.get("high_angle", "Fully extend at the top.")
        if quality == "unstable":
            return rules.get("unstable", "Control your movement.")

        if score < 40:
            return random.choice(self._GENERIC_BAD)

        return "Good effort — focus on completing the full range."
