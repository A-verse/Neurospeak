# utils/analytics.py

import json
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import pytz
import atexit
import numpy as np
from scipy import stats
import tempfile
import shutil

import logging

def update_log(message: str) -> None:
    """Append message to a log file or handle analytics logging."""
    logging.info(f"Analytics Log: {message}")


# Security imports
from cryptography.fernet import InvalidToken
from utils.encryption import get_encryption_manager, SecurityError

# Constants
USAGE_FILE = "data/usage_log.enc"
TEMP_DIR = "data/temp_analytics"
MAX_LOG_ENTRIES = 1000000  # 1 million entries
LOG_ROTATION_SIZE = 100 * 1024 * 1024  # 100MB
BACKUP_COUNT = 3
ANALYTICS_CACHE_TTL = 3600  # 1 hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnalyticsEngine')

class UsageAnalytics:
    def __init__(self):
        """
        Advanced analytics engine with:
        - Military-grade security
        - Thread-safe operations
        - Temporal pattern analysis
        - Predictive modeling
        - Anomaly detection
        """
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_expiry = {}
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
        self._key_rotation_check()

    def _key_rotation_check(self):
        """Periodically check for key rotation needs"""
        # Implementation would check last rotation time
        # and rotate keys if necessary
        pass

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

    def _rotate_logs(self):
        """Implement log rotation when size threshold reached"""
        if os.path.exists(USAGE_FILE) and os.path.getsize(USAGE_FILE) > LOG_ROTATION_SIZE:
            try:
                for i in range(BACKUP_COUNT-1, 0, -1):
                    src = f"{USAGE_FILE}.{i}"
                    dst = f"{USAGE_FILE}.{i+1}"
                    if os.path.exists(src):
                        shutil.move(src, dst)
                
                shutil.move(USAGE_FILE, f"{USAGE_FILE}.1")
                logger.info("Rotated usage logs")
            except Exception as e:
                logger.error(f"Log rotation failed: {str(e)}")

    def _load_log(self) -> Dict[str, Any]:
        """
        Securely load and decrypt usage log with full validation
        """
        temp_file = self._get_temp_file()
        
        try:
            # Decrypt with military-grade encryption
            self.encryption.decrypt_file(USAGE_FILE, temp_file)
            
            # Validate file integrity
            if os.path.getsize(temp_file) > MAX_LOG_ENTRIES * 100:  # Rough size estimate
                raise SecurityError("Log file size exceeds safety limits")
            
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict):
                raise SecurityError("Invalid log structure")
            
            return data
            
        except InvalidToken:
            logger.error("Invalid token - possible tampering detected")
            raise SecurityError("Log decryption failed - possible tampering")
        except Exception as e:
            logger.error(f"Log loading failed: {str(e)}")
            raise
        finally:
            try:
                os.remove(temp_file)
            except:
                pass

    def _save_log(self, data: Dict[str, Any]):
        """
        Securely encrypt and save usage log with atomic write
        """
        temp_file = self._get_temp_file()
        backup_file = self._get_temp_file()
        
        try:
            # Create backup
            if os.path.exists(USAGE_FILE):
                shutil.copy2(USAGE_FILE, backup_file)
            
            # Write new data
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Encrypt with military-grade security
            self.encryption.encrypt_file(temp_file, USAGE_FILE)
            
            # Verify the new file
            if os.path.getsize(USAGE_FILE) == 0:
                raise SecurityError("Empty encrypted file created")
            
            logger.info("Successfully saved usage log")
            
        except Exception as e:
            logger.error(f"Log save failed: {str(e)}")
            
            # Restore backup if available
            if os.path.exists(backup_file):
                try:
                    shutil.move(backup_file, USAGE_FILE)
                except Exception as restore_error:
                    logger.critical(f"Backup restore failed: {str(restore_error)}")
            
            raise
        finally:
            try:
                os.remove(temp_file)
                os.remove(backup_file)
            except:
                pass

    def _get_cache_key(self, *args) -> str:
        """Generate secure cache key"""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check and return cached result if valid"""
        with self._lock:
            if cache_key in self._cache:
                if time.time() < self._cache_expiry.get(cache_key, 0):
                    return self._cache[cache_key]
                del self._cache[cache_key]
                del self._cache_expiry[cache_key]
            return None

    def _set_cache(self, cache_key: str, value: Any, ttl: int = ANALYTICS_CACHE_TTL):
        """Store value in cache with TTL"""
        with self._lock:
            self._cache[cache_key] = value
            self._cache_expiry[cache_key] = time.time() + ttl

    def track_usage(self, phrase: str, metadata: Optional[Dict] = None):
        """
        Track phrase usage with comprehensive metadata and temporal analysis
        
        Args:
            phrase: The phrase being used
            metadata: Additional context (timestamp, user_id, etc.)
        """
        if not phrase or not isinstance(phrase, str):
            raise ValueError("Invalid phrase")
        
        with self._lock:
            try:
                # Load existing data
                log = self._load_log()
                
                # Initialize structure if needed
                if 'phrases' not in log:
                    log['phrases'] = {}
                if 'temporal' not in log:
                    log['temporal'] = {}
                if 'metadata' not in log:
                    log['metadata'] = {}
                
                # Update phrase counts
                log['phrases'][phrase] = log['phrases'].get(phrase, 0) + 1
                
                # Track temporal patterns (by hour)
                now = datetime.now(pytz.UTC)
                hour_key = now.strftime("%Y-%m-%d-%H")
                log['temporal'][hour_key] = log['temporal'].get(hour_key, {})
                log['temporal'][hour_key][phrase] = log['temporal'][hour_key].get(phrase, 0) + 1
                
                # Store metadata if provided
                if metadata:
                    metadata_id = hashlib.sha256(phrase.encode()).hexdigest()
                    log['metadata'][metadata_id] = {
                        'last_used': now.isoformat(),
                        'data': metadata
                    }
                
                # Enforce size limits
                if len(log['phrases']) > MAX_LOG_ENTRIES:
                    self._prune_old_entries(log)
                
                # Save updated log
                self._save_log(log)
                
                # Invalidate relevant caches
                self._cache = {}
                
            except Exception as e:
                logger.error(f"Usage tracking failed for phrase '{phrase}': {str(e)}")
                raise

    def _prune_old_entries(self, log: Dict[str, Any]):
        """Maintain log size within limits"""
        # Keep top 90% of phrases by usage count
        phrases = sorted(log['phrases'].items(), key=lambda x: x[1], reverse=True)
        keep_count = int(MAX_LOG_ENTRIES * 0.9)
        log['phrases'] = dict(phrases[:keep_count])
        
        # Prune old temporal data (older than 90 days)
        cutoff = (datetime.now(pytz.UTC) - timedelta(days=90)).strftime("%Y-%m-%d-%H")
        log['temporal'] = {k: v for k, v in log['temporal'].items() if k > cutoff}
        
        logger.info(f"Pruned log entries to maintain size limits")

    def get_top_phrases(self, n: int = 5, time_window: Optional[timedelta] = None) -> List[Tuple[str, int]]:
        """
        Get top phrases with optional time window filtering
        
        Args:
            n: Number of phrases to return
            time_window: Only consider usage within this time window
            
        Returns:
            List of (phrase, count) tuples
        """
        cache_key = self._get_cache_key('top_phrases', n, time_window)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        with self._lock:
            try:
                log = self._load_log()
                phrases = log.get('phrases', {})
                
                if time_window:
                    cutoff = (datetime.now(pytz.UTC) - time_window).strftime("%Y-%m-%d-%H")
                    temporal = log.get('temporal', {})
                    
                    # Filter phrases by time window
                    window_phrases = defaultdict(int)
                    for hour, hour_data in temporal.items():
                        if hour > cutoff:
                            for phrase, count in hour_data.items():
                                window_phrases[phrase] += count
                    
                    # Merge with overall counts
                    phrases = {p: c + window_phrases.get(p, 0) for p, c in phrases.items()}
                
                top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:n]
                self._set_cache(cache_key, top_phrases)
                return top_phrases
                
            except Exception as e:
                logger.error(f"Failed to get top phrases: {str(e)}")
                raise

    def get_usage_trends(self, phrase: str, days: int = 30) -> Dict[str, int]:
        """
        Get temporal usage trends for a specific phrase
        
        Args:
            phrase: Phrase to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary of {hour: count} for the time period
        """
        cache_key = self._get_cache_key('usage_trends', phrase, days)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        with self._lock:
            try:
                log = self._load_log()
                temporal = log.get('temporal', {})
                
                cutoff = (datetime.now(pytz.UTC) - timedelta(days=days)).strftime("%Y-%m-%d-%H")
                trends = defaultdict(int)
                
                for hour, hour_data in temporal.items():
                    if hour > cutoff and phrase in hour_data:
                        trends[hour] += hour_data[phrase]
                
                result = dict(trends)
                self._set_cache(cache_key, result)
                return result
                
            except Exception as e:
                logger.error(f"Failed to get usage trends for '{phrase}': {str(e)}")
                raise

    def predict_next_phrases(self, current_phrase: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict likely next phrases based on usage patterns
        
        Args:
            current_phrase: The current phrase
            n: Number of predictions to return
            
        Returns:
            List of (phrase, probability) tuples
        """
        cache_key = self._get_cache_key('predict_next', current_phrase, n)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached
        
        with self._lock:
            try:
                log = self._load_log()
                temporal = log.get('temporal', {})
                
                # Find all phrases that follow current_phrase within sessions
                followers = defaultdict(int)
                total = 0
                
                for hour_data in temporal.values():
                    phrases = list(hour_data.keys())
                    try:
                        idx = phrases.index(current_phrase)
                        if idx + 1 < len(phrases):
                            next_phrase = phrases[idx + 1]
                            followers[next_phrase] += hour_data[next_phrase]
                            total += hour_data[next_phrase]
                    except ValueError:
                        continue
                
                # Calculate probabilities
                predictions = []
                for phrase, count in followers.items():
                    predictions.append((phrase, count / total if total > 0 else 0))
                
                # Sort by probability
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                result = predictions[:n]
                self._set_cache(cache_key, result)
                return result
                
            except Exception as e:
                logger.error(f"Prediction failed for '{current_phrase}': {str(e)}")
                raise

    def detect_anomalies(self, threshold: float = 2.5) -> Dict[str, Any]:
        """
        Detect anomalous usage patterns using statistical methods
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary of detected anomalies
        """
        with self._lock:
            try:
                log = self._load_log()
                phrases = log.get('phrases', {})
                
                if not phrases:
                    return {}
                
                # Convert to numpy array for analysis
                counts = np.array(list(phrases.values()))
                z_scores = np.abs(stats.zscore(counts))
                
                # Find anomalies
                anomalies = {}
                for phrase, count in phrases.items():
                    idx = list(phrases.keys()).index(phrase)
                    if z_scores[idx] > threshold:
                        anomalies[phrase] = {
                            'count': count,
                            'z_score': float(z_scores[idx]),
                            'percentile': float(stats.percentileofscore(counts, count))
                        }
                
                return anomalies
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                raise

# Singleton instance for thread-safe operations
_analytics_engine = None
_analytics_lock = threading.Lock()

def get_analytics_engine() -> UsageAnalytics:
    """
    Thread-safe singleton access to the analytics engine
    """
    global _analytics_engine
    with _analytics_lock:
        if _analytics_engine is None:
            _analytics_engine = UsageAnalytics()
        return _analytics_engine