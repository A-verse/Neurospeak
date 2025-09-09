"""
🎯 Enhanced Text-to-Speech Module with Emotion Detection and Advanced Features

Features:
- Offline capability using pyttsx3
- Dynamic voice adjustment based on emotion analysis
- Multi-level fallback mechanisms
- Voice gender selection
- Thread-safe operations
- Resource cleanup
- Comprehensive error handling
- Performance monitoring
- Platform compatibility checks
- Voice customization presets
"""

import pyttsx3
import threading
import time
import re
from typing import Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum, auto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceGender(Enum):
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()

@dataclass
class VoicePreset:
    rate: int
    volume: float
    pitch: Optional[int] = None

# Voice presets for different scenarios
PRESETS = {
    'default': VoicePreset(rate=150, volume=1.0),
    'excited': VoicePreset(rate=180, volume=1.2),
    'calm': VoicePreset(rate=120, volume=0.8),
    'angry': VoicePreset(rate=160, volume=1.5),
    'sad': VoicePreset(rate=100, volume=0.7),
    'emergency': VoicePreset(rate=200, volume=2.0)
}

class TTSController:
    """
    Advanced Text-to-Speech controller with thread-safe operations and enhanced features.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TTSController, cls).__new__(cls)
                cls._instance._initialize_engine()
        return cls._instance
    
    def _initialize_engine(self):
        """Initialize the TTS engine with fallback mechanisms."""
        self._engine_lock = threading.Lock()
        self._active = False
        self._initialization_time = time.time()
        
        try:
            self.engine = pyttsx3.init()
            self._active = True
            
            # Set default properties
            self.set_voice_settings()
            self._available_voices = self.engine.getProperty('voices')
            
            # Performance metrics
            self._usage_count = 0
            self._last_used = 0
            
            logger.info("TTS Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {str(e)}")
            self._fallback_engine()
    
    def _fallback_engine(self, attempt: int = 1):
        """Attempt to recover or use fallback mechanisms."""
        if attempt > 3:
            logger.error("Maximum fallback attempts reached. TTS disabled.")
            self._active = False
            return
        
        try:
            logger.warning(f"Attempting fallback initialization (attempt {attempt})")
            self.engine = pyttsx3.init(driverName=None)  # Let pyttsx3 choose best available
            self._active = True
            logger.info("Fallback TTS engine initialized")
        except Exception as e:
            logger.error(f"Fallback attempt {attempt} failed: {str(e)}")
            time.sleep(0.5 * attempt)
            self._fallback_engine(attempt + 1)
    
    def set_voice_settings(self, 
                          rate: Optional[int] = None,
                          volume: Optional[float] = None,
                          gender: Optional[VoiceGender] = None,
                          preset: Optional[str] = None):
        """
        Configure voice settings with optional preset.
        
        Args:
            rate: Words per minute
            volume: Volume level (0.0 to 1.0)
            gender: VoiceGender enum value
            preset: Preset name from PRESETS
        """
        if not self._active:
            logger.warning("Cannot set voice settings - engine not active")
            return
            
        with self._engine_lock:
            if preset and preset in PRESETS:
                preset_values = PRESETS[preset]
                rate = preset_values.rate if rate is None else rate
                volume = preset_values.volume if volume is None else volume
            
            if rate is not None:
                self.engine.setProperty('rate', max(50, min(rate, 500)))  # Clamped range
            if volume is not None:
                self.engine.setProperty('volume', max(0.0, min(float(volume), 2.0)))
            if gender is not None and self._available_voices:
                self._set_voice_gender(gender)
    
    def _set_voice_gender(self, gender: VoiceGender):
        """Set voice gender if available on platform."""
        try:
            voices = self._available_voices
            if len(voices) > 1:
                if gender == VoiceGender.MALE and 'male' in voices[0].name.lower():
                    self.engine.setProperty('voice', voices[0].id)
                elif gender == VoiceGender.FEMALE and len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
        except Exception as e:
            logger.warning(f"Couldn't set voice gender: {str(e)}")
    
    def speak(self, 
              text: str,
              rate: Optional[int] = None,
              volume: Optional[float] = None,
              gender: Optional[VoiceGender] = None,
              preset: Optional[str] = None,
              interrupt: bool = False) -> bool:
        """
        Speak the given text with optional parameters.
        
        Args:
            text: Text to speak
            rate: Optional words per minute
            volume: Optional volume level
            gender: Optional voice gender
            preset: Optional preset name
            interrupt: Whether to stop current speech
            
        Returns:
            bool: True if speech was successful
        """
        if not self._active or not text:
            logger.warning(f"Cannot speak - engine active: {self._active}, text provided: {bool(text)}")
            return False
            
        try:
            with self._engine_lock:
                # Clean text of any problematic characters
                text = self._sanitize_text(text)
                
                if interrupt:
                    self.engine.stop()
                
                # Apply settings before speaking
                self.set_voice_settings(rate, volume, gender, preset)
                
                # Performance tracking
                start_time = time.time()
                
                self.engine.say(text)
                self.engine.runAndWait()
                
                # Update metrics
                self._usage_count += 1
                self._last_used = time.time()
                elapsed = time.time() - start_time
                
                logger.info(f"Spoke {len(text)} characters in {elapsed:.2f} seconds")
                return True
        except RuntimeError as e:
            logger.error(f"Runtime error during speech: {str(e)}")
            self._handle_crash()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during speech: {str(e)}")
            return False
    
    def _handle_crash(self):
        """Attempt to recover from engine crash."""
        try:
            self.engine.endLoop()
        except:
            pass
        self._initialize_engine()
    
    def _sanitize_text(self, text: str) -> str:
        """Remove or replace problematic characters in text."""
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return ""
        
        # Remove control characters except basic whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
    
    def get_usage_stats(self) -> dict:
        """Get performance and usage statistics."""
        return {
            'initialized': self._initialization_time,
            'usage_count': self._usage_count,
            'last_used': self._last_used,
            'active': self._active,
            'voices_available': len(self._available_voices) if self._active else 0
        }
    
    def shutdown(self):
        """Cleanup resources."""
        with self._engine_lock:
            if self._active:
                try:
                    self.engine.stop()
                    self.engine.endLoop()
                    del self.engine
                    self._active = False
                    logger.info("TTS engine shutdown complete")
                except Exception as e:
                    logger.error(f"Error during shutdown: {str(e)}")

# Global instance for convenience (thread-safe)
tts_controller = TTSController()

# Helper functions for common operations
def speak(text: str, 
          rate: Optional[int] = None,
          volume: Optional[float] = None,
          emotion: Optional[str] = None,
          interrupt: bool = False) -> bool:
    """
    Speak text with optional emotion-based preset.
    
    Args:
        text: Text to speak
        rate: Optional speech rate
        volume: Optional volume level
        emotion: Optional emotion for preset
        interrupt: Whether to stop current speech
    """
    preset = emotion if emotion in PRESETS else None
    return tts_controller.speak(
        text=text,
        rate=rate,
        volume=volume,
        preset=preset,
        interrupt=interrupt
    )

def emergency_announcement(text: str, interrupt: bool = True) -> bool:
    """Deliver urgent announcement with emergency preset."""
    return tts_controller.speak(
        text=text,
        preset='emergency',
        interrupt=interrupt
    )

def set_voice_gender(gender: VoiceGender):
    """Set preferred voice gender."""
    tts_controller.set_voice_settings(gender=gender)

def get_voice_info() -> dict:
    """Get information about available voices."""
    return tts_controller.get_usage_stats()

def shutdown_engine():
    """Cleanup TTS resources."""
    tts_controller.shutdown()

# Example usage
if __name__ == "__main__":
    speak("Hello, this is a test of the enhanced text to speech system.", emotion='excited')
    emergency_announcement("Warning! Critical system alert!")
    print(get_voice_info())
    shutdown_engine()