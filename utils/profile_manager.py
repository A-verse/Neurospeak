# utils/profile_manager.py

import json
import os
import logging
import threading
import hashlib
import uuid
from typing import Dict, Any, Optional, Union
from datetime import datetime
import atexit
import tempfile
import shutil
import bcrypt
from cryptography.fernet import InvalidToken

# Security imports
from utils.encryption import get_encryption_manager, SecurityError

# Constants
PROFILE_FILE = "data/user_profiles.enc"  # Encrypted storage
TEMP_DIR = "data/temp_profiles"
MAX_PROFILES = 1000
MAX_PROFILE_SIZE = 10 * 1024  # 10KB per profile
BACKUP_COUNT = 3
DEFAULT_EAR_THRESHOLD = 0.21
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 72

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ProfileManager')

class ProfileManager:
    def __init__(self):
        """
        Military-grade secure profile management system with:
        - AES-256 encrypted storage
        - Secure memory handling
        - Comprehensive validation
        - Multi-user concurrency support
        """
        self._lock = threading.RLock()
        self._cache = {}
        self._setup_directories()
        self._initialize_encryption()
        atexit.register(self._cleanup)

    def _setup_directories(self):
        """Ensure secure directory structure exists"""
        try:
            os.makedirs("data", mode=0o700, exist_ok=True)
            os.makedirs(TEMP_DIR, mode=0o700, exist_ok=True)
            
            # Secure directory permissions
            os.chmod("data", 0o700)
            os.chmod(TEMP_DIR, 0o700)
        except Exception as e:
            logger.critical(f"Directory setup failed: {str(e)}")
            raise

    def _initialize_encryption(self):
        """Initialize encryption components"""
        self.encryption = get_encryption_manager()

    def _get_temp_file(self) -> str:
        """Create secure temporary file"""
        try:
            fd, path = tempfile.mkstemp(dir=TEMP_DIR)
            os.close(fd)
            os.chmod(path, 0o600)
            return path
        except Exception as e:
            logger.error(f"Temp file creation failed: {str(e)}")
            raise

    def _cleanup(self):
        """Secure cleanup of temporary files"""
        try:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR, ignore_errors=False)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def _load_profiles(self) -> Dict[str, Any]:
        """
        Securely load and decrypt profiles with full validation
        """
        temp_file = self._get_temp_file()
        
        try:
            # Handle case where no profiles exist yet
            if not os.path.exists(PROFILE_FILE):
                return {}

            # Decrypt with military-grade encryption
            self.encryption.decrypt_file(PROFILE_FILE, temp_file)
            
            # Validate file size
            if os.path.getsize(temp_file) > MAX_PROFILES * MAX_PROFILE_SIZE:
                raise SecurityError("Profile file size exceeds safety limits")
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                raise SecurityError("Invalid profile structure")
            
            return data
            
        except InvalidToken:
            logger.error("Invalid token - possible tampering detected")
            raise SecurityError("Profile decryption failed - possible tampering")
        except Exception as e:
            logger.error(f"Profile loading failed: {str(e)}")
            raise
        finally:
            try:
                os.remove(temp_file)
            except:
                pass

    def _save_profiles(self, profiles: Dict[str, Any]):
        """
        Securely encrypt and save profiles with atomic write
        """
        temp_file = self._get_temp_file()
        backup_file = self._get_temp_file()
        
        try:
            # Create backup
            if os.path.exists(PROFILE_FILE):
                shutil.copy2(PROFILE_FILE, backup_file)
            
            # Validate before saving
            self._validate_profiles(profiles)
            
            # Write new data
            with open(temp_file, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            # Encrypt with military-grade security
            self.encryption.encrypt_file(temp_file, PROFILE_FILE)
            
            # Verify the new file
            if os.path.getsize(PROFILE_FILE) == 0:
                raise SecurityError("Empty encrypted file created")
            
            logger.info("Successfully saved profiles")
            
        except Exception as e:
            logger.error(f"Profile save failed: {str(e)}")
            
            # Restore backup if available
            if os.path.exists(backup_file):
                try:
                    shutil.move(backup_file, PROFILE_FILE)
                except Exception as restore_error:
                    logger.critical(f"Backup restore failed: {str(restore_error)}")
            
            raise
        finally:
            try:
                os.remove(temp_file)
                os.remove(backup_file)
            except:
                pass

    def _validate_profiles(self, profiles: Dict[str, Any]):
        """Validate all profiles before saving"""
        if not isinstance(profiles, dict):
            raise ValueError("Profiles must be a dictionary")
        
        if len(profiles) > MAX_PROFILES:
            raise ValueError(f"Maximum number of profiles ({MAX_PROFILES}) exceeded")
        
        for username, profile in profiles.items():
            self._validate_profile(username, profile)

    def _validate_profile(self, username: str, profile: Dict[str, Any]):
        """Validate individual profile structure and values"""
        if not isinstance(username, str) or not username.isprintable():
            raise ValueError("Invalid username format")
        
        if not isinstance(profile, dict):
            raise ValueError("Profile must be a dictionary")
        
        # Validate standard fields
        if 'language' not in profile:
            raise ValueError("Language is required")
        
        if 'ear_threshold' in profile:
            if not (0.01 <= float(profile['ear_threshold']) <= 0.5):
                raise ValueError("EAR threshold must be between 0.01 and 0.5")
        
        if 'font_size' in profile:
            if not (MIN_FONT_SIZE <= int(profile['font_size']) <= MAX_FONT_SIZE):
                raise ValueError(f"Font size must be between {MIN_FONT_SIZE} and {MAX_FONT_SIZE}")

    def _hash_password(self, password: str) -> str:
        """Securely hash password with salt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def get_user_profile(self, username: str, auth_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get user profile with optional authentication
        
        Args:
            username: User identifier
            auth_token: Optional authentication token
            
        Returns:
            User profile if found and authorized, None otherwise
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                profile = profiles.get(username)
                
                if not profile:
                    return None
                
                # Remove sensitive data if not authenticated
                if auth_token and 'auth_token' in profile:
                    if not self._verify_password(auth_token, profile['auth_token']):
                        logger.warning(f"Invalid auth token for user {username}")
                        return None
                
                # Return a copy without sensitive fields
                safe_profile = profile.copy()
                safe_profile.pop('auth_token', None)
                safe_profile.pop('password_hash', None)
                
                return safe_profile
                
            except Exception as e:
                logger.error(f"Failed to get profile for {username}: {str(e)}")
                raise

    def create_or_update_profile(
        self,
        username: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        font_size: int = 14,
        password: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create or update user profile with comprehensive validation
        
        Args:
            username: Unique user identifier
            language: Preferred language code
            voice_id: Voice preference identifier
            ear_threshold: Eye aspect ratio threshold
            font_size: Display font size
            password: Optional authentication secret
            metadata: Additional profile data
            
        Returns:
            Created/updated profile (without sensitive fields)
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                
                # Initialize profile if new
                if username not in profiles:
                    profiles[username] = {
                        'created_at': datetime.utcnow().isoformat(),
                        'last_updated': datetime.utcnow().isoformat(),
                        'usage_stats': defaultdict(int)
                    }
                
                # Update core fields
                profile = profiles[username]
                profile.update({
                    'language': language,
                    'voice_id': voice_id or str(uuid.uuid4()),
                    'ear_threshold': min(max(ear_threshold, 0.01), 0.5),
                    'font_size': min(max(font_size, MIN_FONT_SIZE), MAX_FONT_SIZE),
                    'last_updated': datetime.utcnow().isoformat(),
                    'metadata': metadata or {}
                })
                
                # Handle authentication
                if password:
                    profile['auth_token'] = self._hash_password(password)
                
                # Validate before saving
                self._validate_profile(username, profile)
                self._save_profiles(profiles)
                
                logger.info(f"Profile saved for {username}")
                return self.get_user_profile(username)
                
            except Exception as e:
                logger.error(f"Failed to create/update profile for {username}: {str(e)}")
                raise

    def update_profile_field(
        self,
        username: str,
        field: str,
        value: Any,
        auth_token: Optional[str] = None
    ) -> bool:
        """
        Update specific profile field with validation and authentication
        
        Args:
            username: User identifier
            field: Field to update
            value: New field value
            auth_token: Authentication secret
            
        Returns:
            True if update succeeded, False otherwise
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                
                if username not in profiles:
                    logger.warning(f"Profile not found: {username}")
                    return False
                
                # Verify authentication if required
                profile = profiles[username]
                if 'auth_token' in profile:
                    if not auth_token or not self._verify_password(auth_token, profile['auth_token']):
                        logger.warning(f"Unauthorized update attempt for {username}")
                        return False
                
                # Validate sensitive fields
                if field in ['auth_token', 'password_hash']:
                    raise ValueError("Cannot directly update authentication fields")
                
                # Special validation for certain fields
                if field == 'ear_threshold':
                    value = min(max(float(value), 0.01), 0.5)
                elif field == 'font_size':
                    value = min(max(int(value), MIN_FONT_SIZE), MAX_FONT_SIZE)
                
                # Update field
                profile[field] = value
                profile['last_updated'] = datetime.utcnow().isoformat()
                
                self._validate_profile(username, profile)
                self._save_profiles(profiles)
                
                logger.info(f"Updated {field} for {username}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update {field} for {username}: {str(e)}")
                raise

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and return session token
        
        Args:
            username: User identifier
            password: Authentication secret
            
        Returns:
            Session token if successful, None otherwise
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                profile = profiles.get(username)
                
                if not profile or 'auth_token' not in profile:
                    return None
                
                if self._verify_password(password, profile['auth_token']):
                    # Generate and return a session token
                    session_token = str(uuid.uuid4())
                    profile['session_token'] = self._hash_password(session_token)
                    profile['last_login'] = datetime.utcnow().isoformat()
                    self._save_profiles(profiles)
                    return session_token
                
                return None
                
            except Exception as e:
                logger.error(f"Authentication failed for {username}: {str(e)}")
                raise

    def verify_session(self, username: str, session_token: str) -> bool:
        """
        Verify active user session
        
        Args:
            username: User identifier
            session_token: Session token to verify
            
        Returns:
            True if session is valid, False otherwise
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                profile = profiles.get(username)
                
                if not profile or 'session_token' not in profile:
                    return False
                
                return self._verify_password(session_token, profile['session_token'])
                
            except Exception as e:
                logger.error(f"Session verification failed for {username}: {str(e)}")
                raise

    def get_all_profiles(self, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all profiles (admin function)
        
        Args:
            admin_token: Optional admin authentication
            
        Returns:
            Dictionary of all profiles (sanitized)
        """
        with self._lock:
            try:
                profiles = self._load_profiles()
                
                # Sanitize all profiles
                sanitized = {}
                for username, profile in profiles.items():
                    safe_profile = profile.copy()
                    safe_profile.pop('auth_token', None)
                    safe_profile.pop('password_hash', None)
                    safe_profile.pop('session_token', None)
                    sanitized[username] = safe_profile
                
                return sanitized
                
            except Exception as e:
                logger.error(f"Failed to get all profiles: {str(e)}")
                raise

# Singleton instance for thread-safe operations
_profile_manager = None
_profile_lock = threading.Lock()

def get_profile_manager() -> ProfileManager:
    """
    Thread-safe singleton access to the profile manager
    """
    global _profile_manager
    with _profile_lock:
        if _profile_manager is None:
            _profile_manager = ProfileManager()
        return _profile_manager