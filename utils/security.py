"""
Security Manager - Handles encryption, authentication, and security features.
"""

import os
import logging
import base64
import hashlib
import getpass
import json
import secrets
from pathlib import Path
from datetime import datetime, timedelta

# Try importing cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

class SecurityManager:
    """Handles security operations for J.A.R.V.I.S."""
    
    def __init__(self, config=None):
        """Initialize the Security Manager.
        
        Args:
            config: Optional configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Security settings
        self.security_dir = Path.home() / '.jarvis' / 'security'
        self.security_dir.mkdir(parents=True, exist_ok=True)
        
        # Keys and tokens
        self.key_file = self.security_dir / 'key.enc'
        self.session_token = None
        self.session_expiry = None
        
        # Feature availability
        self.encryption_available = CRYPTOGRAPHY_AVAILABLE
        if not self.encryption_available:
            self.logger.warning("Cryptography library not available. Encryption disabled.")
        
        # Initialize encryption key
        self.encryption_key = self._load_or_create_key()
        
        self.logger.info("Security Manager initialized")
    
    def _load_or_create_key(self):
        """Load existing encryption key or create a new one.
        
        Returns:
            Encryption key or None if encryption is not available
        """
        if not self.encryption_available:
            return None
        
        try:
            if not self.key_file.exists():
                # Generate a new key
                key = Fernet.generate_key()
                
                # Save it to the key file (in a real system, this should be better secured)
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                
                self.logger.info("Generated new encryption key")
                return key
            else:
                # Load existing key
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                
                self.logger.debug("Loaded existing encryption key")
                return key
                
        except Exception as e:
            self.logger.error(f"Error with encryption key: {e}")
            return None
    
    def encrypt(self, data):
        """Encrypt data.
        
        Args:
            data: String or bytes to encrypt
            
        Returns:
            Encrypted bytes or None if encryption failed
        """
        if not self.encryption_available or not self.encryption_key:
            self.logger.warning("Encryption not available")
            return None
        
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Create Fernet cipher with the key
            cipher = Fernet(self.encryption_key)
            
            # Encrypt the data
            encrypted_data = cipher.encrypt(data)
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt(self, encrypted_data):
        """Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted bytes or None if decryption failed
        """
        if not self.encryption_available or not self.encryption_key:
            self.logger.warning("Decryption not available")
            return None
        
        try:
            # Create Fernet cipher with the key
            cipher = Fernet(self.encryption_key)
            
            # Decrypt the data
            decrypted_data = cipher.decrypt(encrypted_data)
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return None
    
    def hash_password(self, password, salt=None):
        """Hash a password with a salt.
        
        Args:
            password: Password string
            salt: Optional salt bytes (generated if None)
            
        Returns:
            Tuple of (hash, salt) or None if failed
        """
        try:
            # Generate salt if not provided
            if salt is None:
                salt = os.urandom(16)
            
            # Hash the password
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000  # Number of iterations
            )
            
            return (password_hash, salt)
            
        except Exception as e:
            self.logger.error(f"Password hashing error: {e}")
            return None
    
    def verify_password(self, password, password_hash, salt):
        """Verify a password against a hash.
        
        Args:
            password: Password string to verify
            password_hash: Stored password hash
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Hash the provided password with the same salt
            new_hash, _ = self.hash_password(password, salt)
            
            # Compare the hashes
            return new_hash == password_hash
            
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    def generate_token(self, expires_in=3600):
        """Generate a session token.
        
        Args:
            expires_in: Token expiry in seconds
            
        Returns:
            Session token string
        """
        # Generate a random token
        token = secrets.token_urlsafe(32)
        
        # Set expiry time
        self.session_token = token
        self.session_expiry = datetime.now() + timedelta(seconds=expires_in)
        
        self.logger.debug(f"Generated new session token, expires in {expires_in} seconds")
        
        return token
    
    def validate_token(self, token):
        """Validate a session token.
        
        Args:
            token: Token to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        # Check if token matches and is not expired
        if (self.session_token == token and 
            self.session_expiry and 
            datetime.now() < self.session_expiry):
            return True
            
        return False
    
    def encrypt_file(self, file_path, output_path=None):
        """Encrypt a file.
        
        Args:
            file_path: Path to the file to encrypt
            output_path: Path for the encrypted file (or None to use file_path.enc)
            
        Returns:
            Path to encrypted file or None if failed
        """
        if not self.encryption_available or not self.encryption_key:
            self.logger.warning("File encryption not available")
            return None
        
        try:
            # Default output path
            if output_path is None:
                output_path = f"{file_path}.enc"
            
            # Read the file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Encrypt the data
            encrypted_data = self.encrypt(data)
            
            if encrypted_data:
                # Write encrypted data to output file
                with open(output_path, 'wb') as f:
                    f.write(encrypted_data)
                
                self.logger.info(f"Encrypted {file_path} to {output_path}")
                return output_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"File encryption error: {e}")
            return None
    
    def decrypt_file(self, encrypted_file_path, output_path=None):
        """Decrypt an encrypted file.
        
        Args:
            encrypted_file_path: Path to the encrypted file
            output_path: Path for the decrypted file (or None for default)
            
        Returns:
            Path to decrypted file or None if failed
        """
        if not self.encryption_available or not self.encryption_key:
            self.logger.warning("File decryption not available")
            return None
        
        try:
            # Default output path (remove .enc extension if present)
            if output_path is None:
                output_path = encrypted_file_path
                if output_path.endswith('.enc'):
                    output_path = output_path[:-4]
                else:
                    output_path = f"{output_path}.dec"
            
            # Read the encrypted file
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt the data
            decrypted_data = self.decrypt(encrypted_data)
            
            if decrypted_data:
                # Write decrypted data to output file
                with open(output_path, 'wb') as f:
                    f.write(decrypted_data)
                
                self.logger.info(f"Decrypted {encrypted_file_path} to {output_path}")
                return output_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"File decryption error: {e}")
            return None
    
    def secure_store(self, key, value):
        """Store a value securely.
        
        Args:
            key: Key for the stored value
            value: Value to store (string)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.encryption_available:
            self.logger.warning("Secure storage not available")
            return False
        
        try:
            # Path for the secure store
            store_file = self.security_dir / 'secure_store.enc'
            
            # Load existing store if it exists
            store = {}
            if store_file.exists():
                with open(store_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.decrypt(encrypted_data)
                if decrypted_data:
                    store = json.loads(decrypted_data.decode('utf-8'))
            
            # Add or update the value
            store[key] = value
            
            # Encrypt and save the updated store
            encrypted_data = self.encrypt(json.dumps(store).encode('utf-8'))
            
            if encrypted_data:
                with open(store_file, 'wb') as f:
                    f.write(encrypted_data)
                
                self.logger.debug(f"Stored secure value for key: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Secure store error: {e}")
            return False
    
    def secure_retrieve(self, key):
        """Retrieve a value from secure storage.
        
        Args:
            key: Key for the stored value
            
        Returns:
            Stored value or None if not found
        """
        if not self.encryption_available:
            self.logger.warning("Secure retrieval not available")
            return None
        
        try:
            # Path for the secure store
            store_file = self.security_dir / 'secure_store.enc'
            
            # Check if store exists
            if not store_file.exists():
                self.logger.warning("Secure store does not exist")
                return None
            
            # Load and decrypt the store
            with open(store_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.decrypt(encrypted_data)
            if not decrypted_data:
                return None
            
            store = json.loads(decrypted_data.decode('utf-8'))
            
            # Return the value if it exists
            if key in store:
                self.logger.debug(f"Retrieved secure value for key: {key}")
                return store[key]
            
            self.logger.warning(f"Key not found in secure store: {key}")
            return None
            
        except Exception as e:
            self.logger.error(f"Secure retrieve error: {e}")
            return None
