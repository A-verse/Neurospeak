#!/usr/bin/env python3
"""
NeuroSpeak - Gesture-controlled communication aid
Rule-based gesture classifier + Thread-safe UI
"""

import time, logging
from typing import Dict, Any, Optional
from utils.camera_utils import Camera
from ai.gesture_classifier import classify_gesture
from ai.blink_detector import BlinkDetector
from ai.phrase_predictor import PhrasePredictor
from ai.emotion_mapper import get_emotion, get_voice_settings
from audio.text_to_speech import speak
from utils.analytics import update_log
from ui.display_output import init_display, show_phrase
from audio.voice_selector import apply_saved_voice

MIN_FRAME_INTERVAL = 0.1
DEBOUNCE_TIME = 2.0
LOG_FILE = "neurospeak.log"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    )
    logging.getLogger().setLevel(logging.INFO)

def process_input_frame(frame: Any, blink_detector: BlinkDetector, logger) -> Optional[Dict[str, Any]]:
    try:
        gesture = classify_gesture(frame)
        if gesture:
            return {"type": "gesture", "value": gesture}

        blink_detector.process_frame(frame)
        event = blink_detector.get_blink_event()
        if event and event[0].name == "DOUBLE":
            return {"type": "blink", "value": "double_blink"}

    except Exception as e:
        logger.error(f"Frame processing error: {e}")
    return None

def generate_output(input_data: Dict[str, Any], predictor: PhrasePredictor, logger):
    try:
        suggestions, method = predictor.suggest_phrases(input_data["value"])
        if not suggestions:
            return
        phrase = suggestions[0]
        emotion = get_emotion(phrase)
        voice_cfg = get_voice_settings(emotion)

        show_phrase(phrase, speaker=input_data.get("value"))
        speak(phrase, rate=voice_cfg["rate"], volume=voice_cfg["volume"])
        update_log(phrase)
        predictor.record_phrase_usage(phrase, gesture=input_data.get("value"))
        logger.info(f"Output: {phrase} | Emotion: {emotion} | Method: {method.name}")
    except Exception as e:
        logger.error(f"Output generation error: {e}")

def main():
    setup_logging()
    logger = logging.getLogger("NeuroSpeak")
    logger.info("🧠 NeuroSpeak started")

    blink_detector = BlinkDetector()
    predictor = PhrasePredictor()
    init_display()
    apply_saved_voice()

    try:
        with Camera() as cam:
            last_output = 0
            while True:
                now = time.time()
                if now - last_output < MIN_FRAME_INTERVAL:
                    time.sleep(0.05)
                    continue

                frame = cam.get_frame()
                if frame is None or frame.size == 0:
                    continue

                input_data = process_input_frame(frame, blink_detector, logger)
                if not input_data:
                    continue

                if now - last_output < DEBOUNCE_TIME:
                    continue

                generate_output(input_data, predictor, logger)
                last_output = now

    except KeyboardInterrupt:
        logger.info("👋 Shutting down NeuroSpeak...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("✅ NeuroSpeak shutdown complete")

if __name__ == "__main__":
    main()
