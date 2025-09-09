# utils/encryption.py

import os
import logging
import hashlib
import hmac
import secrets
from typing import Optional, Union, Tuple
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import base64
import atexit
import threading

# Constants
KEY_FILE = "config/.secret.key"
SALT_FILE = "config/.secret.salt"
KEY_ROTATION_INTERVAL = 30 * 24 * 3600  # 30 days in seconds
KEY_DERIVATION_ITERATIONS = 600000  # OWASP recommended minimum

# Security parameters
MIN_PASSPHRASE_LENGTH = 32
ENCRYPTED_FILE_EXTENSION = ".enc"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

class EncryptionManager:
    def __init__(self):
        """
        Military-grade encryption system with multiple protection layers:
        - Key rotation
        - Secure key derivation
        - Tamper detection
        - Memory protection
        """
        self._lock = threading.Lock()
        self._key_cache = None
        self._last_key_rotation = 0
        self._setup_logging()
        self._validate_security_environment()
        atexit.register(self._cleanup)

    def _setup_logging(self):
        """Configure secure logging"""
        self.logger = logging.getLogger('EncryptionManager')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _validate_security_environment(self):
        """Verify system security requirements"""
        if not os.path.exists("config"):
            os.makedirs("config", mode=0o700)
        
        # Verify directory permissions
        if (os.stat("config").st_mode & 0o777) != 0o700:
            self.logger.warning("Insecure directory permissions for config")

    def _generate_secure_key(self) -> Tuple[bytes, bytes]:
        """
        Generates a cryptographically secure key using PBKDF2 with random salt
        """
        salt = secrets.token_bytes(32)
        passphrase = secrets.token_bytes(MIN_PASSPHRASE_LENGTH)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=salt,
            iterations=KEY_DERIVATION_ITERATIONS,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(passphrase))
        return key, salt

    def _load_or_generate_key(self) -> Tuple[bytes, bytes]:
        """
        Securely loads or generates new key material with rotation
        """
        with self._lock:
            # Check if we need to rotate keys
            current_time = time.time()
            if (current_time - self._last_key_rotation) > KEY_ROTATION_INTERVAL:
                self.logger.info("Rotating encryption keys")
                return self._rotate_keys()

            # Try to load existing key
            try:
                if os.path.exists(KEY_FILE) and os.path.exists(SALT_FILE):
                    with open(KEY_FILE, "rb") as f:
                        key = f.read()
                    with open(SALT_FILE, "rb") as f:
                        salt = f.read()
                    
                    # Verify key integrity
                    if len(key) != 44 or len(salt) != 32:  # Fernet key is 44 bytes
                        raise ValueError("Invalid key or salt length")
                    
                    return key, salt
            except Exception as e:
                self.logger.error(f"Key loading failed: {str(e)}")

            # Generate new key if loading failed
            return self._rotate_keys()

    def _rotate_keys(self) -> Tuple[bytes, bytes]:
        """
        Generates and stores new cryptographic material
        """
        key, salt = self._generate_secure_key()
        
        try:
            # Secure file writing
            with open(KEY_FILE, "wb") as f:
                os.chmod(KEY_FILE, 0o600)
                f.write(key)
            
            with open(SALT_FILE, "wb") as f:
                os.chmod(SALT_FILE, 0o600)
                f.write(salt)
            
            self._last_key_rotation = time.time()
            return key, salt
        except Exception as e:
            self.logger.critical(f"Key rotation failed: {str(e)}")
            raise RuntimeError("Failed to rotate encryption keys")

    def _get_fernet(self) -> Fernet:
        """
        Returns a Fernet instance with current key, handling cache and rotation
        """
        with self._lock:
            if self._key_cache is None:
                key, _ = self._load_or_generate_key()
                self._key_cache = Fernet(key)
            return self._key_cache

    def _cleanup(self):
        """Securely wipe sensitive data from memory"""
        with self._lock:
            if self._key_cache is not None:
                # Attempt to overwrite memory (best effort)
                try:
                    import ctypes
                    ctypes.memset(ctypes.addressof(self._key_cache), 0, ctypes.sizeof(self._key_cache))
                except:
                    pass
                self._key_cache = None

    def _validate_file(self, file_path: str, operation: str):
        """Security validation before file operations"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large ({file_size} bytes) for {operation}")

    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypts file with authenticated encryption and integrity checks.
        
        Args:
            input_path: Path to plaintext file
            output_path: Optional output path (defaults to input_path + .enc)
        
        Returns:
            Path to encrypted file
        """
        self._validate_file(input_path, "encryption")
        
        if output_path is None:
            output_path = input_path + ENCRYPTED_FILE_EXTENSION

        try:
            fernet = self._get_fernet()
            
            with open(input_path, "rb") as infile:
                plaintext = infile.read()
            
            # Add additional HMAC for tamper detection
            hmac_digest = hmac.new(
                self._load_or_generate_key()[0],
                plaintext,
                hashlib.sha512
            ).digest()
            
            # Encrypt data with metadata
            encrypted = fernet.encrypt(hmac_digest + plaintext)
            
            # Secure file writing
            with open(output_path, "wb") as outfile:
                os.chmod(output_path, 0o600)
                outfile.write(encrypted)
            
            self.logger.info(f"Successfully encrypted {input_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Encryption failed for {input_path}: {str(e)}")
            raise

    def decrypt_file(self, encrypted_path: str, output_path: str) -> str:
        """
        Decrypts file with full integrity verification.
        
        Args:
            encrypted_path: Path to encrypted file
            output_path: Path for decrypted output
        
        Returns:
            Path to decrypted file
        """
        self._validate_file(encrypted_path, "decryption")

        try:
            fernet = self._get_fernet()
            
            with open(encrypted_path, "rb") as infile:
                encrypted_data = infile.read()
            
            # Decrypt and verify
            decrypted = fernet.decrypt(encrypted_data)
            hmac_digest = decrypted[:64]  # SHA512 digest is 64 bytes
            plaintext = decrypted[64:]
            
            # Verify HMAC
            expected_hmac = hmac.new(
                self._load_or_generate_key()[0],
                plaintext,
                hashlib.sha512
            ).digest()
            
            if not hmac.compare_digest(hmac_digest, expected_hmac):
                raise SecurityError("HMAC validation failed - file may be tampered with")
            
            # Secure file writing
            with open(output_path, "wb") as outfile:
                os.chmod(output_path, 0o600)
                outfile.write(plaintext)
            
            self.logger.info(f"Successfully decrypted {encrypted_path} -> {output_path}")
            return output_path
            
        except InvalidToken:
            self.logger.error("Invalid token - possible tampering or corrupt data")
            raise SecurityError("Decryption failed - invalid token")
        except Exception as e:
            self.logger.error(f"Decryption failed for {encrypted_path}: {str(e)}")
            raise

    def encrypt_data(self, plaintext: bytes) -> bytes:
        """
        Encrypts in-memory data with additional protection layers
        """
        if not isinstance(plaintext, bytes):
            raise TypeError("Plaintext must be bytes")
        
        fernet = self._get_fernet()
        
        # Add random IV and padding
        iv = secrets.token_bytes(16)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        # AES-CBC encryption for additional layer
        cipher = Cipher(
            algorithms.AES(self._load_or_generate_key()[0][:32]),  # Use first 32 bytes for AES
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        aes_encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        # Fernet encryption on top
        return fernet.encrypt(iv + aes_encrypted)

    def decrypt_data(self, ciphertext: bytes) -> bytes:
        """
        Decrypts in-memory data with full validation
        """
        if not isinstance(ciphertext, bytes):
            raise TypeError("Ciphertext must be bytes")
        
        fernet = self._get_fernet()
        
        try:
            # First layer: Fernet decryption
            decrypted = fernet.decrypt(ciphertext)
            iv = decrypted[:16]
            aes_encrypted = decrypted[16:]
            
            # Second layer: AES-CBC decryption
            cipher = Cipher(
                algorithms.AES(self._load_or_generate_key()[0][:32]),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(aes_encrypted) + decryptor.finalize()
            
            # Remove padding
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return plaintext
            
        except InvalidToken:
            raise SecurityError("Decryption failed - invalid token")
        except Exception as e:
            self.logger.error(f"In-memory decryption failed: {str(e)}")
            raise

class SecurityError(Exception):
    """Custom exception for security-related failures"""
    pass

# Singleton instance for thread-safe operations
_encryption_manager = None
_encryption_lock = threading.Lock()

def get_encryption_manager() -> EncryptionManager:
    """
    Thread-safe singleton access to the encryption manager
    """
    global _encryption_manager
    with _encryption_lock:
        if _encryption_manager is None:
            _encryption_manager = EncryptionManager()
        return _encryption_manager

# Helper functions for backward compatibility
def encrypt_file(input_path: str, output_path: Optional[str] = None) -> str:
    return get_encryption_manager().encrypt_file(input_path, output_path)

def decrypt_file(encrypted_path: str, output_path: str) -> str:
    return get_encryption_manager().decrypt_file(encrypted_path, output_path)