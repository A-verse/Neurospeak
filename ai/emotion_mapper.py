"""
ai/emotion_mapper.py - Advanced Emotional Mapping System

Enhanced Features:
1. Multi-language emotion mapping
2. Context-aware emotional analysis
3. Gesture-to-emotion mapping
4. Emotion intensity levels
5. Fallback systems and error handling
6. Dynamic emotion adjustment
7. Security and validation layers
8. Future-proof architecture
"""

import logging
from typing import Dict, Union, Optional
from dataclasses import dataclass
import re
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels for emotional data access"""
    PUBLIC = auto()
    PRIVATE = auto()
    CONFIDENTIAL = auto()

@dataclass
class VoiceSettings:
    """Structured voice parameters for TTS"""
    rate: int
    volume: float
    pitch: float = 1.0
    pause_duration: float = 0.1

class EmotionMapper:
    """Comprehensive emotion mapping system with security and context awareness"""
    
    def __init__(self):
        # Core emotion mappings
        self._phrase_to_emotion = self._initialize_phrase_mappings()
        self._gesture_to_emotion = self._initialize_gesture_mappings()
        self._emotion_settings = self._initialize_emotion_settings()
        self._language_support = self._initialize_language_support()
        
        # Security parameters
        self._security_level = SecurityLevel.PUBLIC
        self._allowed_domains = ["emotional_mapping", "voice_synthesis"]
        
        # Context parameters
        self._context_history = []
        self._max_context_length = 10

    @staticmethod
    def _initialize_phrase_mappings() -> Dict[str, Dict[str, str]]:
        """Initialize phrase to emotion mappings with intensity levels"""
        return {
            # English phrases
            "en": {
                "Yes": {"emotion": "neutral", "intensity": 1.0},
                "No": {"emotion": "neutral", "intensity": 1.0},
                "I need help": {"emotion": "urgent", "intensity": 1.5},
                "Thank you": {"emotion": "grateful", "intensity": 1.2},
                "Thanks a lot": {"emotion": "grateful", "intensity": 1.5},
                "I'm happy": {"emotion": "joyful", "intensity": 1.3},
                "I'm so happy": {"emotion": "joyful", "intensity": 1.7},
                "I'm sad": {"emotion": "sad", "intensity": 1.3},
                "I'm very sad": {"emotion": "sad", "intensity": 1.8},
                "Goodbye": {"emotion": "gentle", "intensity": 1.0},
                "See you later": {"emotion": "gentle", "intensity": 1.2},
                "Hello": {"emotion": "cheerful", "intensity": 1.1},
                "Hi there": {"emotion": "cheerful", "intensity": 1.3},
                "Watch out": {"emotion": "alert", "intensity": 1.8},
                "Danger": {"emotion": "fearful", "intensity": 2.0},
                "I love this": {"emotion": "loving", "intensity": 1.7},
                "That's amazing": {"emotion": "excited", "intensity": 1.6},
                "I'm sorry": {"emotion": "apologetic", "intensity": 1.4},
                "Please": {"emotion": "pleading", "intensity": 1.3},
                "Congratulations": {"emotion": "proud", "intensity": 1.5},
            },
            # Add other languages here following the same structure
        }

    @staticmethod
    def _initialize_gesture_mappings() -> Dict[str, Dict[str, str]]:
        """Initialize gesture to emotion mappings"""
        return {
            "nod": {"emotion": "agreeing", "intensity": 1.2},
            "head shake": {"emotion": "disagreeing", "intensity": 1.2},
            "thumbs up": {"emotion": "approving", "intensity": 1.4},
            "thumbs down": {"emotion": "disapproving", "intensity": 1.4},
            "wave": {"emotion": "friendly", "intensity": 1.1},
            "clap": {"emotion": "applauding", "intensity": 1.6},
            "point": {"emotion": "directing", "intensity": 1.3},
            "shrug": {"emotion": "uncertain", "intensity": 1.0},
        }

    @staticmethod
    def _initialize_emotion_settings() -> Dict[str, VoiceSettings]:
        """Initialize detailed voice settings for each emotion"""
        return {
            "neutral": VoiceSettings(rate=150, volume=1.0),
            "urgent": VoiceSettings(rate=180, volume=1.0, pitch=1.1),
            "joyful": VoiceSettings(rate=170, volume=1.2, pitch=1.15),
            "sad": VoiceSettings(rate=120, volume=0.8, pitch=0.9),
            "grateful": VoiceSettings(rate=140, volume=1.0, pitch=1.05),
            "gentle": VoiceSettings(rate=130, volume=0.9, pitch=0.95),
            "cheerful": VoiceSettings(rate=160, volume=1.1, pitch=1.1),
            "alert": VoiceSettings(rate=190, volume=1.3, pitch=1.2),
            "fearful": VoiceSettings(rate=200, volume=1.4, pitch=1.3),
            "loving": VoiceSettings(rate=145, volume=1.1, pitch=1.05),
            "excited": VoiceSettings(rate=175, volume=1.3, pitch=1.2),
            "apologetic": VoiceSettings(rate=125, volume=0.9, pitch=0.95),
            "pleading": VoiceSettings(rate=135, volume=0.95, pitch=0.98),
            "proud": VoiceSettings(rate=155, volume=1.15, pitch=1.1),
            "agreeing": VoiceSettings(rate=150, volume=1.0, pitch=1.0),
            "disagreeing": VoiceSettings(rate=150, volume=1.0, pitch=0.95),
            "approving": VoiceSettings(rate=160, volume=1.1, pitch=1.05),
            "disapproving": VoiceSettings(rate=140, volume=1.0, pitch=0.9),
            "friendly": VoiceSettings(rate=155, volume=1.05, pitch=1.02),
            "applauding": VoiceSettings(rate=165, volume=1.25, pitch=1.1),
            "directing": VoiceSettings(rate=170, volume=1.1, pitch=1.05),
            "uncertain": VoiceSettings(rate=130, volume=0.95, pitch=0.98),
        }

    @staticmethod
    def _initialize_language_support() -> Dict[str, str]:
        """Initialize supported languages"""
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            # Add more languages as needed
        }

    def _validate_input(self, input_str: str) -> bool:
        """Validate input against injection patterns"""
        if not isinstance(input_str, str):
            return False
            
        injection_patterns = [
            r"<script.*?>.*?</script>",
            r"[\;\|\&\$\>\<]",
            r"\/\*.*?\*\/",
            r"--",
            r"\b(?:DROP|ALTER|CREATE|INSERT|UPDATE|DELETE|TRUNCATE)\b",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                logger.warning(f"Potential injection detected in input: {input_str}")
                return False
                
        return True

    def _sanitize_input(self, input_str: str) -> str:
        """Sanitize input string"""
        return re.sub(r"[^\w\s.,!?']", "", input_str.strip())

    def _update_context(self, context: str):
        """Maintain context history for emotional continuity"""
        self._context_history.append(context)
        if len(self._context_history) > self._max_context_length:
            self._context_history.pop(0)

    def get_emotion(
        self, 
        input_data: str, 
        input_type: str = "phrase", 
        language: str = "en",
        context: Optional[str] = None
    ) -> Dict[str, Union[str, float]]:
        """
        Get emotion mapping for input with context awareness
        
        Args:
            input_data: The phrase or gesture to analyze
            input_type: Either 'phrase' or 'gesture'
            language: Language code (default 'en')
            context: Optional context string
            
        Returns:
            Dictionary with emotion and intensity
            
        Raises:
            ValueError: For invalid input types or security violations
        """
        # Security and input validation
        if not self._validate_input(input_data):
            raise ValueError("Invalid input detected")
            
        if input_type not in ["phrase", "gesture"]:
            raise ValueError("input_type must be either 'phrase' or 'gesture'")
            
        if language not in self._language_support:
            language = "en"  # Default to English
            
        sanitized_input = self._sanitize_input(input_data)
        
        # Update context if provided
        if context:
            self._update_context(context)
            
        # Get emotion mapping
        try:
            if input_type == "phrase":
                # Check exact matches first
                if language in self._phrase_to_emotion:
                    for phrase, mapping in self._phrase_to_emotion[language].items():
                        if sanitized_input.lower() == phrase.lower():
                            return mapping
                
                # Check partial matches with context awareness
                for phrase, mapping in self._phrase_to_emotion[language].items():
                    if phrase.lower() in sanitized_input.lower():
                        # Adjust intensity based on context
                        adjusted_mapping = mapping.copy()
                        if context and "urgent" in context.lower():
                            adjusted_mapping["intensity"] = min(2.0, adjusted_mapping["intensity"] * 1.3)
                        return adjusted_mapping
                        
            elif input_type == "gesture":
                for gesture, mapping in self._gesture_to_emotion.items():
                    if gesture.lower() in sanitized_input.lower():
                        return mapping
                        
        except Exception as e:
            logger.error(f"Error processing emotion mapping: {str(e)}")
            
        # Fallback to neutral with default intensity
        return {"emotion": "neutral", "intensity": 1.0}

    def get_voice_settings(
        self, 
        emotion: str, 
        intensity: float = 1.0,
        security_check: bool = True
    ) -> VoiceSettings:
        """
        Get voice settings for emotion with intensity adjustment
        
        Args:
            emotion: Emotion label
            intensity: Emotion intensity (default 1.0)
            security_check: Whether to perform security validation
            
        Returns:
            VoiceSettings object with adjusted parameters
            
        Raises:
            ValueError: For invalid emotion or security violations
        """
        if security_check and self._security_level == SecurityLevel.CONFIDENTIAL:
            raise ValueError("Access denied for current security level")
            
        if not isinstance(intensity, (int, float)) or intensity <= 0:
            intensity = 1.0
            
        # Clamp intensity to reasonable bounds
        intensity = max(0.5, min(2.5, intensity))
        
        try:
            base_settings = self._emotion_settings.get(emotion.lower(), self._emotion_settings["neutral"])
            
            # Adjust settings based on intensity
            adjusted_settings = VoiceSettings(
                rate=int(base_settings.rate * (1 + (intensity - 1) * 0.2)),
                volume=base_settings.volume * intensity,
                pitch=base_settings.pitch * (1 + (intensity - 1) * 0.1),
                pause_duration=max(0.05, base_settings.pause_duration / intensity)
            )
            
            return adjusted_settings
            
        except Exception as e:
            logger.error(f"Error generating voice settings: {str(e)}")
            return self._emotion_settings["neutral"]

    def set_security_level(self, level: SecurityLevel):
        """Set security level for emotion mapping"""
        if not isinstance(level, SecurityLevel):
            raise ValueError("Invalid security level")
        self._security_level = level

    def add_custom_mapping(
        self, 
        input_data: str, 
        emotion: str, 
        intensity: float = 1.0,
        input_type: str = "phrase",
        language: str = "en"
    ):
        """
        Add custom emotion mapping
        
        Args:
            input_data: Phrase or gesture to map
            emotion: Emotion label
            intensity: Emotion intensity
            input_type: 'phrase' or 'gesture'
            language: Language code
            
        Raises:
            ValueError: For invalid parameters
        """
        if not self._validate_input(input_data) or not self._validate_input(emotion):
            raise ValueError("Invalid input data")
            
        if input_type == "phrase":
            if language not in self._phrase_to_emotion:
                self._phrase_to_emotion[language] = {}
            self._phrase_to_emotion[language][input_data] = {
                "emotion": emotion,
                "intensity": max(0.1, min(2.5, intensity))
            }
        elif input_type == "gesture":
            self._gesture_to_emotion[input_data] = {
                "emotion": emotion,
                "intensity": max(0.1, min(2.5, intensity))
            }
        else:
            raise ValueError("Invalid input_type")

# Singleton instance for global use
emotion_mapper = EmotionMapper()

# Legacy functions for backward compatibility
def get_emotion(phrase: str) -> str:
    """Legacy function - returns just the emotion string"""
    mapping = emotion_mapper.get_emotion(phrase)
    return mapping["emotion"]

def get_voice_settings(emotion: str) -> Dict[str, Union[int, float]]:
    """Legacy function - returns basic voice settings"""
    settings = emotion_mapper.get_voice_settings(emotion)
    return {"rate": settings.rate, "volume": settings.volume}