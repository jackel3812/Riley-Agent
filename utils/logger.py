"""
Logging Utility for J.A.R.V.I.S.
Provides a centralized logging system for all J.A.R.V.I.S. components.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime

# Default log directory
DEFAULT_LOG_DIR = os.path.expanduser("~/.jarvis/logs")

def setup_logger(level=logging.INFO, log_file=None, console=True):
    """Set up the J.A.R.V.I.S. logger.
    
    Args:
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_file (str, optional): Path to log file. Defaults to None.
        console (bool, optional): Whether to log to console. Defaults to True.
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided or use default location
    if log_file is None:
        # Create default log directory if it doesn't exist
        if not os.path.exists(DEFAULT_LOG_DIR):
            try:
                os.makedirs(DEFAULT_LOG_DIR)
            except Exception as e:
                print(f"Failed to create log directory: {e}")
                return root_logger
        
        # Create log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(DEFAULT_LOG_DIR, f"jarvis-{timestamp}.log")
    
    try:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to set up file logging: {e}")
    
    # Log initial message
    root_logger.debug("Logger initialized")
    
    return root_logger

def get_logger(name):
    """Get a logger for a specific component.
    
    Args:
        name (str): Logger name, typically __name__
    
    Returns:
        logging.Logger: Logger for the component
    """
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """Adapter that adds context information to log messages."""
    
    def __init__(self, logger, extra=None):
        """Initialize logger adapter.
        
        Args:
            logger (logging.Logger): Logger to adapt
            extra (dict, optional): Extra context information. Defaults to None.
        """
        self.logger = logger
        self.extra = extra or {}
    
    def process(self, msg, kwargs):
        """Process the log message.
        
        Args:
            msg (str): Log message
            kwargs (dict): Additional parameters
        
        Returns:
            tuple: Processed message and kwargs
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs