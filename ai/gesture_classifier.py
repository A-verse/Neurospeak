#!/usr/bin/env python3
"""
Gesture Classifier - Final Clean Version (~400 lines)
- Pure OpenCV rule-based
- Adjustable screen size
- Debug overlay (FPS, confidence)
- Integrated smoothing for stability
"""

import cv2
import numpy as np
import time
import logging
from collections import deque
from typing import Optional, Tuple, Dict

# ✅ Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GestureClassifier")

# ✅ Gesture Labels
GESTURES = {
    0: "NONE",
    1: "OPEN_PALM",
    2: "FIST",
    3: "THUMBS_UP",
    4: "THUMBS_DOWN",
    5: "POINTING",
    6: "RAISED_HAND",
    7: "HEAD_NOD",
    8: "HEAD_SHAKE"
}

# ✅ Config
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5
MIN_CONTOUR_AREA = 5000
SKIN_LOWER = np.array([0, 30, 60], dtype=np.uint8)
SKIN_UPPER = np.array([20, 150, 255], dtype=np.uint8)

# ✅ Haarcascade for face
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class GestureClassifier:
    def __init__(self, smoothing_window: int = SMOOTHING_WINDOW):
        self.prediction_history = deque(maxlen=smoothing_window)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_face_center = None
        self.debug_enabled = True

    # ====== FPS Calculation ======
    def _calc_fps(self):
        self.frame_count += 1
        if self.frame_count >= 10:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    # ====== Preprocessing ======
    def _preprocess(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.medianBlur(mask, 7)
        return mask

    def _find_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None
        return largest

    # ====== Finger Counting ======
    def _count_fingers(self, contour, frame):
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c + 1e-6))
            if angle <= np.pi / 2 and d > 10000:
                finger_count += 1
                cv2.circle(frame, far, 5, (0, 255, 0), -1)
        return finger_count

    # ====== Hand Gesture Detection ======
    def _detect_hand_gesture(self, frame) -> Tuple[int, float]:
        mask = self._preprocess(frame)
        contour = self._find_largest_contour(mask)
        if contour is None:
            return 0, 0.0

        hull_points = cv2.convexHull(contour)
        cv2.drawContours(frame, [hull_points], -1, (255, 0, 0), 2)

        finger_count = self._count_fingers(contour, frame)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Rule-based Classification
        if finger_count == 0:
            if aspect_ratio > 1.2:
                return 2, 0.9  # Fist
            else:
                return 6, 0.85  # Raised hand
        elif finger_count >= 4:
            return 1, 0.95  # Open palm
        elif finger_count == 1:
            if y < frame.shape[0] // 3:
                return 3, 0.85  # Thumbs up
            elif y > 2 * frame.shape[0] // 3:
                return 4, 0.85  # Thumbs down
            else:
                return 5, 0.8  # Pointing
        else:
            return 1, 0.6

    # ====== Head Movement Detection ======
    def _detect_head_movement(self, frame_gray):
        faces = FACE_CASCADE.detectMultiScale(frame_gray, 1.3, 5)
        if len(faces) == 0:
            self.last_face_center = None
            return None

        x, y, w, h = faces[0]
        center = (x + w // 2, y + h // 2)

        gesture = None
        if self.last_face_center:
            dx = center[0] - self.last_face_center[0]
            dy = center[1] - self.last_face_center[1]
            if abs(dx) > 20 and abs(dx) > abs(dy):
                gesture = 8  # Head shake
            elif abs(dy) > 20 and abs(dy) > abs(dx):
                gesture = 7  # Head nod

        self.last_face_center = center
        return gesture

    # ====== Smoothing ======
    def _smooth_prediction(self, pred_id: int) -> int:
        self.prediction_history.append(pred_id)
        if len(self.prediction_history) < SMOOTHING_WINDOW:
            return pred_id
        return max(set(self.prediction_history), key=self.prediction_history.count)

    # ====== Main Process ======
    def process_frame(self, frame) -> Tuple[Optional[int], Dict]:
        self._calc_fps()
        debug = {"fps": self.fps, "confidence": 0.0, "hands": False, "face": False}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gesture_id, conf = self._detect_hand_gesture(frame)
        if gesture_id != 0 and conf >= CONFIDENCE_THRESHOLD:
            gesture_id = self._smooth_prediction(gesture_id)
            debug["confidence"] = conf
            debug["hands"] = True
            return gesture_id, debug

        head_gesture = self._detect_head_movement(gray)
        if head_gesture:
            gesture_id = self._smooth_prediction(head_gesture)
            debug["confidence"] = 0.9
            debug["face"] = True
            return gesture_id, debug

        return 0, debug

    # ====== Debug Drawing ======
    def draw_debug(self, frame, gesture_id, debug):
        cv2.putText(frame, f"FPS: {debug['fps']:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if gesture_id != 0:
            cv2.putText(frame, f"Gesture: {GESTURES[gesture_id]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {debug['confidence']:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame


# ✅ External Wrapper for main.py
_classifier = GestureClassifier()

def classify_gesture(frame):
    gesture_id, debug = _classifier.process_frame(frame)
    if gesture_id == 0:
        return None
    return GESTURES[gesture_id]
