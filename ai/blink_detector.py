#!/usr/bin/env python3
# ai/blink_detector.py
"""
Blink Detector - Pure OpenCV Rule-Based (No Dlib)
Detects blinks using Haar Cascade face & eye detection, EAR calculation,
temporal smoothing, and classification into SINGLE / DOUBLE / LONG.
"""

import cv2
import numpy as np
import time
from scipy.spatial import distance
from collections import deque
from typing import List, Tuple, Optional, Dict
from enum import Enum, auto
import logging
import threading
import os
import sys

# ============ Logging ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blink_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============ Constants ============
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

DEFAULT_EAR_THRESHOLD = 0.21
DEFAULT_CONSEC_FRAMES = 3
DEFAULT_EAR_SMOOTHING_WINDOW = 5
DEFAULT_BLINK_COOLDOWN = 0.5  # seconds between blinks

CALIBRATION_FRAMES = 30

# ============ Enums ============
class BlinkType(Enum):
    SINGLE = auto()
    DOUBLE = auto()
    LONG = auto()

class EyeState(Enum):
    OPEN = auto()
    CLOSED = auto()
    TRANSITION = auto()

# ============ Blink Detector ============
class BlinkDetector:
    def __init__(self):
        """Initialize Haar cascades and state variables"""
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

        # Thresholds and config
        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.consec_frames = DEFAULT_CONSEC_FRAMES
        self.ear_smoothing_window = DEFAULT_EAR_SMOOTHING_WINDOW
        self.blink_cooldown = DEFAULT_BLINK_COOLDOWN

        # State tracking
        self.ear_history = deque(maxlen=self.ear_smoothing_window)
        self.eye_state = EyeState.OPEN
        self.blink_start_time = 0
        self.last_blink_time = 0
        self.blink_sequence = []
        self.blink_count = 0

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Thread safety
        self.lock = threading.Lock()

        # Face/Eye detection flags
        self.face_detected = False

    # ===== FPS Update =====
    def _update_fps(self):
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    # ===== EAR Calculation =====
    def _calculate_ear_from_bbox(self, eye_bbox: Tuple[int, int, int, int]) -> float:
        """
        Approximate Eye Aspect Ratio using eye bounding box height/width.
        """
        (ex, ey, ew, eh) = eye_bbox
        if ew == 0:
            return 0.0
        ear = eh / float(ew)
        return ear

    def _smooth_ear(self, ear: float) -> float:
        self.ear_history.append(ear)
        return np.mean(self.ear_history)

    # ===== Eye Detection =====
    def _detect_eyes(self, frame_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            self.face_detected = False
            return []

        self.face_detected = True
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
        roi_gray = frame_gray[y:y+h, x:x+w]

        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        eye_bboxes = []
        for (ex, ey, ew, eh) in eyes:
            eye_bboxes.append((x+ex, y+ey, ew, eh))
        return eye_bboxes

    # ===== Blink Classification =====
    def _determine_blink_type(self, duration: float) -> Optional[BlinkType]:
        if duration < 0.4:
            return BlinkType.SINGLE
        elif 0.4 <= duration < 1.0:
            return BlinkType.LONG
        return None

    # ===== State Update =====
    def _update_eye_state(self, ear: float):
        smoothed_ear = self._smooth_ear(ear)

        if smoothed_ear < self.ear_threshold:
            if self.eye_state == EyeState.OPEN:
                self.eye_state = EyeState.TRANSITION
                self.blink_start_time = time.time()
            elif self.eye_state == EyeState.TRANSITION:
                self.eye_state = EyeState.CLOSED
        else:
            if self.eye_state == EyeState.CLOSED:
                duration = time.time() - self.blink_start_time
                blink_type = self._determine_blink_type(duration)
                if blink_type:
                    with self.lock:
                        self.blink_sequence.append((blink_type, time.time()))
                self.eye_state = EyeState.OPEN
            elif self.eye_state == EyeState.TRANSITION:
                self.eye_state = EyeState.OPEN

    # ===== Frame Processing =====
    def process_frame(self, frame: np.ndarray) -> Dict:
        debug_info = {
            'fps': self.fps,
            'ear': 0.0,
            'face_detected': self.face_detected,
            'eye_state': self.eye_state.name,
            'threshold': self.ear_threshold,
            'blink_count': self.blink_count
        }

        self._update_fps()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self._detect_eyes(frame_gray)

        if eyes:
            ear_values = [self._calculate_ear_from_bbox(e) for e in eyes]
            if ear_values:
                ear = np.mean(ear_values)
                self._update_eye_state(ear)
                debug_info.update({
                    'ear': ear,
                    'face_detected': True,
                    'smoothed_ear': self._smooth_ear(ear)
                })
        else:
            debug_info['face_detected'] = False

        return debug_info

    # ===== Blink Events =====
    def get_blink_event(self) -> Optional[Tuple[BlinkType, float]]:
        with self.lock:
            if not self.blink_sequence:
                return None
            now = time.time()
            self.blink_sequence = [
                (bt, t) for bt, t in self.blink_sequence
                if now - t < 2.0 and now - self.last_blink_time > self.blink_cooldown
            ]
            if self.blink_sequence:
                event = self.blink_sequence.pop(0)
                self.last_blink_time = event[1]
                self.blink_count += 1
                return event
        return None

    # ===== Debug Overlay =====
    def draw_debug_info(self, frame: np.ndarray, debug_info: Dict) -> np.ndarray:
        cv2.putText(frame, f"FPS: {debug_info['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if debug_info['face_detected']:
            cv2.putText(frame, f"EAR: {debug_info['ear']:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {debug_info['eye_state']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {debug_info['threshold']:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face/eyes detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Blinks: {debug_info['blink_count']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    # ===== Calibration =====
    def calibrate(self, frames: List[np.ndarray]) -> bool:
        ear_values = []
        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self._detect_eyes(frame_gray)
            if eyes:
                ear_vals = [self._calculate_ear_from_bbox(e) for e in eyes]
                if ear_vals:
                    ear_values.append(np.mean(ear_vals))
        if ear_values:
            self.ear_threshold = np.mean(ear_values) * 0.8
            logger.info(f"Auto-calibrated EAR threshold to {self.ear_threshold:.3f}")
            return True
        return False

# ============ Test Runner ============
def main():
    detector = BlinkDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Cannot open camera")
        sys.exit(1)

    # Calibration
    calibration_frames = []
    for _ in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if ret:
            calibration_frames.append(frame)
    detector.calibrate(calibration_frames)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame. Exiting...")
                break

            debug_info = detector.process_frame(frame)
            blink_event = detector.get_blink_event()
            if blink_event:
                blink_type, timestamp = blink_event
                logger.info(f"Detected blink: {blink_type.name}")
                cv2.putText(frame, f"BLINK: {blink_type.name}", (frame.shape[1]//2-100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            frame = detector.draw_debug_info(frame, debug_info)
            cv2.imshow("Blink Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
