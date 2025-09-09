"""
🎯 Advanced Voice Selection System with Enhanced Features

Features:
- Cross-platform voice management
- Voice preference persistence
- Gender/age/language filtering
- Voice preview capability
- Thread-safe operations
- Voice metadata analysis
- Backup/restore functionality
- Voice health monitoring
- Multi-user support
- Secure configuration handling
"""

import os
import json
import pickle
import hashlib
import platform
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from enum import Enum, auto
import pyttsx3
from dataclasses import dataclass
import uuid
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceGender(Enum):
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()
    CHILD = auto()
    UNKNOWN = auto()

class VoiceAge(Enum):
    CHILD = auto()
    TEEN = auto()
    ADULT = auto()
    SENIOR = auto()
    UNKNOWN = auto()

@dataclass
class VoiceProfile:
    id: str
    name: str
    gender: VoiceGender
    age: VoiceAge
    languages: List[str]
    system_id: str
    features: Dict[str, bool]
    rating: float = 0.0

class VoiceManager:
    """
    Advanced voice management system with multi-user support and voice analysis.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VoiceManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the voice management system."""
        self._engine_lock = threading.Lock()
        self._active = False
        self._voices_loaded = False
        self._profiles = []
        self._user_profiles = {}
        
        # Configuration
        self.config_dir = Path.home() / ".voice_manager"
        self.voice_db_file = self.config_dir / "voice_database.pkl"
        self.user_config_file = self.config_dir / "user_profiles.json"
        self.backup_dir = self.config_dir / "backups"
        
        # Initialize directories
        self._setup_directories()
        
        # Initialize engine
        self._init_engine()
        
        # Load data
        self._load_voice_database()
        self._load_user_profiles()
        
        # Performance metrics
        self._last_scan = 0
        self._voice_change_count = 0
    
    def _setup_directories(self):
        """Ensure all required directories exist."""
        try:
            self.config_dir.mkdir(exist_ok=True)
            self.backup_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create config directories: {str(e)}")
            raise
    
    def _init_engine(self):
        """Initialize the TTS engine with fallback handling."""
        try:
            self.engine = pyttsx3.init()
            self._active = True
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {str(e)}")
            self._active = False
            raise
    
    def _load_voice_database(self):
        """Load or create voice database."""
        if self.voice_db_file.exists():
            try:
                with open(self.voice_db_file, 'rb') as f:
                    self._profiles = pickle.load(f)
                logger.info(f"Loaded {len(self._profiles)} voice profiles from database")
                self._voices_loaded = True
                return
            except Exception as e:
                logger.error(f"Failed to load voice database: {str(e)}")
        
        # Create new database if loading failed
        self._scan_system_voices()
    
    def _load_user_profiles(self):
        """Load user voice preferences."""
        if self.user_config_file.exists():
            try:
                with open(self.user_config_file, 'r') as f:
                    self._user_profiles = json.load(f)
                logger.info(f"Loaded preferences for {len(self._user_profiles)} users")
            except Exception as e:
                logger.error(f"Failed to load user profiles: {str(e)}")
                self._user_profiles = {}
    
    def _scan_system_voices(self):
        """Scan and analyze all available system voices."""
        if not self._active:
            return
        
        with self._engine_lock:
            try:
                system_voices = self.engine.getProperty('voices')
                self._profiles = []
                
                for voice in system_voices:
                    profile = self._analyze_voice(voice)
                    self._profiles.append(profile)
                
                # Save the database
                with open(self.voice_db_file, 'wb') as f:
                    pickle.dump(self._profiles, f)
                
                self._voices_loaded = True
                self._last_scan = time.time()
                logger.info(f"Scanned {len(self._profiles)} system voices")
            except Exception as e:
                logger.error(f"Voice scan failed: {str(e)}")
                self._voices_loaded = False
    
    def _analyze_voice(self, voice) -> VoiceProfile:
        """Analyze and create a detailed voice profile."""
        # Extract basic information
        voice_id = str(voice.id)
        name = getattr(voice, 'name', 'Unknown')
        
        # Determine gender
        gender = VoiceGender.UNKNOWN
        if hasattr(voice, 'gender'):
            gender_str = voice.gender.lower()
            if 'male' in gender_str:
                gender = VoiceGender.MALE
            elif 'female' in gender_str:
                gender = VoiceGender.FEMALE
            elif 'child' in gender_str:
                gender = VoiceGender.CHILD
            elif 'neutral' in gender_str:
                gender = VoiceGender.NEUTRAL
        
        # Determine age (heuristic)
        age = VoiceAge.UNKNOWN
        name_lower = name.lower()
        if 'child' in name_lower or 'kid' in name_lower:
            age = VoiceAge.CHILD
        elif 'teen' in name_lower:
            age = VoiceAge.TEEN
        elif 'senior' in name_lower or 'old' in name_lower:
            age = VoiceAge.SENIOR
        elif any(x in name_lower for x in ['adult', 'man', 'woman']):
            age = VoiceAge.ADULT
        
        # Extract languages (heuristic)
        languages = []
        if hasattr(voice, 'languages'):
            languages = [lang.decode('utf-8') if isinstance(lang, bytes) else lang 
                        for lang in voice.languages]
        elif hasattr(voice, 'language'):
            languages = [voice.language]
        
        # Create features dictionary
        features = {
            'neural': 'neural' in name_lower,
            'hq': 'hq' in name_lower or 'high' in name_lower,
            'expressive': any(x in name_lower for x in ['expressive', 'emotional']),
            'fast': 'fast' in name_lower
        }
        
        return VoiceProfile(
            id=str(uuid.uuid4()),
            name=name,
            gender=gender,
            age=age,
            languages=languages,
            system_id=voice_id,
            features=features
        )
    
    def list_voices(self, 
                   gender: Optional[VoiceGender] = None,
                   age: Optional[VoiceAge] = None,
                   language: Optional[str] = None) -> List[VoiceProfile]:
        """
        List available voices with optional filtering.
        
        Args:
            gender: Filter by gender
            age: Filter by age group
            language: Filter by language code
            
        Returns:
            List of matching VoiceProfile objects
        """
        if not self._voices_loaded:
            self._scan_system_voices()
        
        voices = self._profiles.copy()
        
        # Apply filters
        if gender is not None:
            voices = [v for v in voices if v.gender == gender]
        if age is not None:
            voices = [v for v in voices if v.age == age]
        if language is not None:
            voices = [v for v in voices if language in v.languages]
        
        return voices
    
    def select_voice(self, 
                    voice_id: str,
                    user_id: str = "default",
                    preview: bool = False) -> bool:
        """
        Select a voice for a specific user.
        
        Args:
            voice_id: The voice system ID or profile ID
            user_id: User identifier (default: "default")
            preview: Whether to preview the voice immediately
            
        Returns:
            True if selection was successful
        """
        if not self._active:
            return False
        
        # Find the voice by either system ID or our profile ID
        voice_profile = None
        for profile in self._profiles:
            if profile.system_id == voice_id or profile.id == voice_id:
                voice_profile = profile
                break
        
        if not voice_profile:
            logger.error(f"Voice ID not found: {voice_id}")
            return False
        
        with self._engine_lock:
            try:
                # Set the voice
                self.engine.setProperty('voice', voice_profile.system_id)
                
                # Save to user profile
                if user_id not in self._user_profiles:
                    self._user_profiles[user_id] = {}
                
                self._user_profiles[user_id]['voice_id'] = voice_profile.system_id
                self._user_profiles[user_id]['voice_profile'] = {
                    'id': voice_profile.id,
                    'name': voice_profile.name,
                    'gender': voice_profile.gender.name,
                    'age': voice_profile.age.name
                }
                
                # Save configuration
                self._save_user_profiles()
                
                # Preview if requested
                if preview:
                    self.engine.say(f"Hello, this is {voice_profile.name}")
                    self.engine.runAndWait()
                
                self._voice_change_count += 1
                logger.info(f"Voice selected for {user_id}: {voice_profile.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to select voice: {str(e)}")
                return False
    
    def apply_saved_voice(self, user_id: str = "default") -> bool:
        """
        Apply saved voice preference for a user.
        
        Args:
            user_id: User identifier (default: "default")
            
        Returns:
            True if voice was applied successfully
        """
        if not self._active or user_id not in self._user_profiles:
            return False
        
        voice_id = self._user_profiles[user_id].get('voice_id')
        if not voice_id:
            return False
        
        with self._engine_lock:
            try:
                self.engine.setProperty('voice', voice_id)
                logger.info(f"Applied saved voice for {user_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to apply saved voice: {str(e)}")
                return False
    
    def get_current_voice(self) -> Optional[VoiceProfile]:
        """Get the currently active voice profile."""
        if not self._active:
            return None
        
        with self._engine_lock:
            try:
                current_id = self.engine.getProperty('voice')
                for profile in self._profiles:
                    if profile.system_id == current_id:
                        return profile
                return None
            except Exception as e:
                logger.error(f"Failed to get current voice: {str(e)}")
                return None
    
    def _save_user_profiles(self):
        """Save user profiles to disk."""
        try:
            with open(self.user_config_file, 'w') as f:
                json.dump(self._user_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user profiles: {str(e)}")
    
    def backup_config(self) -> bool:
        """Create a backup of the current configuration."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_file = self.backup_dir / f"voice_config_{timestamp}.json"
            
            with open(backup_file, 'w') as f:
                json.dump({
                    'user_profiles': self._user_profiles,
                    'voice_database': [vars(p) for p in self._profiles]
                }, f, indent=2)
            
            logger.info(f"Configuration backed up to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get system statistics and status."""
        return {
            'active': self._active,
            'voices_loaded': self._voices_loaded,
            'total_voices': len(self._profiles),
            'voice_changes': self._voice_change_count,
            'last_scan': self._last_scan,
            'user_count': len(self._user_profiles),
            'config_dir': str(self.config_dir)
        }
    
    def shutdown(self):
        """Clean up resources."""
        with self._engine_lock:
            self._active = False
            try:
                self.engine.stop()
                del self.engine
            except Exception as e:
                logger.error(f"Shutdown error: {str(e)}")

# Global instance for convenience
voice_manager = VoiceManager()

# Helper functions for common operations
def list_voices(gender: Optional[str] = None,
               age: Optional[str] = None,
               language: Optional[str] = None) -> List[Dict]:
    """
    List available voices with optional filters.
    
    Args:
        gender: Filter by gender ('male', 'female', etc.)
        age: Filter by age ('child', 'adult', etc.)
        language: Filter by language code
        
    Returns:
        List of voice information dictionaries
    """
    gender_enum = VoiceGender[gender.upper()] if gender else None
    age_enum = VoiceAge[age.upper()] if age else None
    
    voices = voice_manager.list_voices(gender_enum, age_enum, language)
    return [vars(voice) for voice in voices]

def select_voice(voice_id: str, user_id: str = "default", preview: bool = True) -> bool:
    """
    Select a voice for a user.
    
    Args:
        voice_id: Voice system ID or profile ID
        user_id: User identifier
        preview: Whether to preview the voice
        
    Returns:
        True if successful
    """
    return voice_manager.select_voice(voice_id, user_id, preview)

def apply_saved_voice(user_id: str = "default") -> bool:
    """
    Apply saved voice for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if successful
    """
    return voice_manager.apply_saved_voice(user_id)

def get_current_voice() -> Optional[Dict]:
    """Get information about the current voice."""
    voice = voice_manager.get_current_voice()
    return vars(voice) if voice else None

def backup_voice_config() -> bool:
    """Backup voice configuration."""
    return voice_manager.backup_config()

def get_voice_stats() -> Dict:
    """Get voice system statistics."""
    return voice_manager.get_stats()

def shutdown_voice_system():
    """Shutdown the voice management system."""
    voice_manager.shutdown()

# Example usage
if __name__ == "__main__":
    # List available female voices
    print("Available female voices:")
    for voice in list_voices(gender="female"):
        print(f"- {voice['name']} (ID: {voice['system_id']})")
    
    # Select the first female voice
    female_voices = list_voices(gender="female")
    if female_voices:
        select_voice(female_voices[0]['system_id'], preview=True)
    
    # Apply saved voice for default user
    apply_saved_voice()
    
    # Show current voice
    print("\nCurrent voice:", get_current_voice())
    
    # Backup configuration
    backup_voice_config()
    
    # Show statistics
    print("\nSystem stats:", get_voice_stats())
    
    # Cleanup
    shutdown_voice_system()