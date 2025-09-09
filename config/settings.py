"""
NeuroSpeak Configuration Module

Centralized settings for the NeuroSpeak application with improved organization,
type safety, and environment awareness.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Type aliases for better code clarity
PathLike = Union[str, Path]
Numeric = Union[int, float]

# --- Environment Configuration ---
class Environment:
    """Runtime environment detection and configuration"""
    PRODUCTION = os.getenv("ENV", "development") == "production"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# --- Path Configuration ---
class Paths:
    """Centralized path management with environment awareness"""
    BASE_DIR = Path(__file__).parent.parent
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    PHRASE_FILE = DATA_DIR / "phrases.csv"
    USER_PROFILE_FILE = DATA_DIR / "user_profiles.json"
    USAGE_LOG_FILE = DATA_DIR / "usage_log.enc"
    
    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    GESTURE_MODEL = MODELS_DIR / "gesture_model.pkl"
    SHAPE_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
    
    # Config paths
    CONFIG_DIR = BASE_DIR / "config"
    VOICE_KEY_FILE = CONFIG_DIR / ".secret.key"
    
    @classmethod
    def ensure_directories_exist(cls) -> None:
        """Create required directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.CONFIG_DIR.mkdir(exist_ok=True)

# Initialize paths
Paths.ensure_directories_exist()

# --- Application Settings ---
class AppSettings:
    """Core application settings with environment overrides"""
    # Language settings
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
    
    # Text-to-speech settings
    DEFAULT_VOICE_RATE = int(os.getenv("DEFAULT_VOICE_RATE", 150))
    DEFAULT_VOICE_VOLUME = float(os.getenv("DEFAULT_VOICE_VOLUME", 1.0))
    DEFAULT_EMOTION = os.getenv("DEFAULT_EMOTION", "neutral")
    
    # Camera settings
    CAMERA_ID = int(os.getenv("CAMERA_ID", 0))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
    
    # UI settings
    DEFAULT_FONT_SIZE = int(os.getenv("DEFAULT_FONT_SIZE", 16))
    UI_THEME = os.getenv("UI_THEME", "light").lower()
    THEMES = ["light", "dark", "high_contrast"]
    
    @classmethod
    def validate_settings(cls) -> None:
        """Validate configuration values"""
        if cls.UI_THEME not in cls.THEMES:
            raise ValueError(f"Invalid UI_THEME: {cls.UI_THEME}. Must be one of {cls.THEMES}")
        if not 0 <= cls.DEFAULT_VOICE_VOLUME <= 1.0:
            raise ValueError("DEFAULT_VOICE_VOLUME must be between 0 and 1")

# Validate app settings
AppSettings.validate_settings()

# --- AI Model Parameters ---
class AIModelParams:
    """Parameters for AI/ML models"""
    # Blink detection parameters
    EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.21))
    BLINK_CONSEC_FRAMES = int(os.getenv("BLINK_CONSEC_FRAMES", 3))
    
    # Gesture recognition confidence threshold
    GESTURE_CONFIDENCE_THRESHOLD = float(os.getenv("GESTURE_CONFIDENCE_THRESHOLD", 0.7))
    
    # Emotion detection parameters
    EMOTION_UPDATE_INTERVAL = float(os.getenv("EMOTION_UPDATE_INTERVAL", 2.0))

# --- Runtime Configuration ---
class RuntimeConfig:
    """Dynamic settings that may change during runtime"""
    # Can be modified during runtime
    current_language: str = AppSettings.DEFAULT_LANGUAGE
    current_theme: str = AppSettings.UI_THEME
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export current runtime configuration as dictionary"""
        return {
            "language": cls.current_language,
            "theme": cls.current_theme,
            "debug": Environment.DEBUG,
            "production": Environment.PRODUCTION
        }

# --- Configuration Export ---
def get_config() -> Dict[str, Any]:
    """Get complete configuration as dictionary"""
    return {
        "environment": {
            "production": Environment.PRODUCTION,
            "debug": Environment.DEBUG
        },
        "paths": {
            "data_dir": str(Paths.DATA_DIR),
            "models_dir": str(Paths.MODELS_DIR),
            "config_dir": str(Paths.CONFIG_DIR)
        },
        "app_settings": {
            "language": AppSettings.DEFAULT_LANGUAGE,
            "voice_rate": AppSettings.DEFAULT_VOICE_RATE,
            "voice_volume": AppSettings.DEFAULT_VOICE_VOLUME,
            "camera_settings": {
                "camera_id": AppSettings.CAMERA_ID,
                "frame_width": AppSettings.FRAME_WIDTH,
                "frame_height": AppSettings.FRAME_HEIGHT
            },
            "ui_settings": {
                "font_size": AppSettings.DEFAULT_FONT_SIZE,
                "theme": AppSettings.UI_THEME
            }
        },
        "ai_params": {
            "ear_threshold": AIModelParams.EAR_THRESHOLD,
            "blink_frames": AIModelParams.BLINK_CONSEC_FRAMES,
            "gesture_confidence": AIModelParams.GESTURE_CONFIDENCE_THRESHOLD
        },
        "runtime": RuntimeConfig.to_dict()
    }

# Example usage:
if __name__ == "__main__":
    import json
    print(json.dumps(get_config(), indent=2))