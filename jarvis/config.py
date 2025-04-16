"""
Configuration Manager for J.A.R.V.I.S.
Handles loading, saving, and accessing configuration settings.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for J.A.R.V.I.S."""
    
    def __init__(self, config_file=None):
        """Initialize the configuration manager.
        
        Args:
            config_file (str, optional): Path to a configuration file. Defaults to None.
        """
        self.config_dir = Path(os.path.expanduser("~/.jarvis"))
        self.config_file = config_file or os.path.join(self.config_dir, "config.json")
        self.config = self._load_default_config()
        
        # Ensure config directory exists
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir)
                logger.info(f"Created configuration directory at {self.config_dir}")
            except Exception as e:
                logger.warning(f"Failed to create config directory: {e}")
        
        # Load saved config if it exists
        if os.path.exists(self.config_file):
            self._load_config()
        else:
            self._save_config()
            logger.info(f"Created new configuration file at {self.config_file}")
    
    def _load_default_config(self):
        """Load the default configuration.
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            "general": {
                "user_name": "sir",
                "debug_mode": False,
                "log_level": "INFO"
            },
            "ai": {
                "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
                "model": "gpt-4o",  # The newest OpenAI model
                "temperature": 0.7,
                "max_tokens": 150
            },
            "voice": {
                "enabled": True,
                "voice_type": "male",
                "rate": 1.0,
                "volume": 1.0
            },
            "database": {
                "db_path": os.path.expanduser("~/.jarvis/jarvis.db"),
                "db_url": os.environ.get("DATABASE_URL", "sqlite:///jarvis.db")
            },
            "interfaces": {
                "web_enabled": True,
                "console_enabled": True,
                "gui_enabled": False
            },
            "security": {
                "auth_required": False,
                "session_timeout": 3600,  # 1 hour
                "api_rate_limit": 100
            },
            "features": {
                "adaptive_learning": True,
                "energy_core_simulation": True,
                "system_monitoring": True
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                saved_config = json.load(f)
                
            # Update config with saved values while preserving default structure
            self._update_config_recursively(self.config, saved_config)
            logger.debug(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _update_config_recursively(self, target, source):
        """Recursively update configuration while preserving structure.
        
        Args:
            target (dict): Target configuration dictionary to update
            source (dict): Source configuration dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursively(target[key], value)
            elif key in target:
                target[key] = value
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.debug(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get(self, section, key=None, default=None):
        """Get a configuration value.
        
        Args:
            section (str): Configuration section
            key (str, optional): Configuration key within section. Defaults to None.
            default (any, optional): Default value if key doesn't exist. Defaults to None.
        
        Returns:
            any: Configuration value or default if not found
        """
        if section in self.config:
            if key is None:
                return self.config[section]
            elif key in self.config[section]:
                return self.config[section][key]
        return default
    
    def set(self, section, key, value):
        """Set a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key within section
            value (any): Value to set
        
        Returns:
            bool: Success flag
        """
        if section not in self.config:
            self.config[section] = {}
        
        # Check if value changed
        if section in self.config and key in self.config[section] and self.config[section][key] == value:
            return True
        
        self.config[section][key] = value
        self._save_config()
        return True
    
    def get_openai_api_key(self):
        """Get the OpenAI API key from environment or config.
        
        Returns:
            str: API key or empty string if not found
        """
        # First check environment variable (higher priority)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # If not in environment, check config
        if not api_key and self.config.get("ai", {}).get("openai_api_key"):
            api_key = self.config["ai"]["openai_api_key"]
        
        return api_key