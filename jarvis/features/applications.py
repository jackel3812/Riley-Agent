"""
Applications Manager - Handles launching and managing applications.
"""

import os
import sys
import logging
import subprocess
import webbrowser
from pathlib import Path

class ApplicationManager:
    """Manages application launching and interaction."""
    
    def __init__(self):
        """Initialize the Application Manager."""
        self.logger = logging.getLogger(__name__)
        self.os_type = sys.platform
        self.app_paths = {}
        self._discover_applications()
        
        self.logger.info("Application Manager initialized")
    
    def _discover_applications(self):
        """Discover installed applications on the system."""
        self.logger.debug("Discovering applications...")
        
        # Default paths to search for applications
        if self.os_type == 'win32':
            # Windows
            search_paths = [
                os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files')),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')),
                os.path.join(os.environ.get('APPDATA', ''), 'Microsoft', 'Windows', 'Start Menu', 'Programs')
            ]
            
            # Common Windows applications
            self.app_paths = {
                'browser': 'microsoft-edge:',
                'chrome': 'chrome',
                'firefox': 'firefox',
                'word': 'winword',
                'excel': 'excel',
                'powerpoint': 'powerpnt',
                'notepad': 'notepad',
                'calculator': 'calc'
            }
            
        elif self.os_type == 'darwin':
            # macOS
            search_paths = [
                '/Applications',
                os.path.expanduser('~/Applications')
            ]
            
            # Common macOS applications
            self.app_paths = {
                'browser': 'open -a Safari',
                'chrome': 'open -a "Google Chrome"',
                'firefox': 'open -a Firefox',
                'word': 'open -a "Microsoft Word"',
                'excel': 'open -a "Microsoft Excel"',
                'powerpoint': 'open -a "Microsoft PowerPoint"',
                'notes': 'open -a Notes',
                'calculator': 'open -a Calculator'
            }
            
        else:
            # Linux
            search_paths = [
                '/usr/bin',
                '/usr/local/bin',
                os.path.expanduser('~/.local/bin')
            ]
            
            # Common Linux applications
            self.app_paths = {
                'browser': 'xdg-open https://www.google.com',
                'chrome': 'google-chrome',
                'firefox': 'firefox',
                'terminal': 'gnome-terminal',
                'calculator': 'gnome-calculator',
                'files': 'nautilus'
            }
        
        # Add more applications based on discovered files
        # This is just a basic implementation; a more comprehensive solution
        # would involve parsing desktop files, registry, etc.
        pass
    
    def get_available_applications(self):
        """Get list of available applications.
        
        Returns:
            List of application names
        """
        return list(self.app_paths.keys())
    
    def launch_application(self, app_name):
        """Launch an application by name.
        
        Args:
            app_name: Name of the application to launch
            
        Returns:
            True if successful, False otherwise
        """
        app_name = app_name.lower()
        
        self.logger.debug(f"Attempting to launch {app_name}")
        
        # Special case for opening URLs
        if app_name in ['browser', 'web', 'internet']:
            return self.open_url('https://www.google.com')
        
        if app_name in self.app_paths:
            cmd = self.app_paths[app_name]
            return self._execute_command(cmd)
        
        # Try to launch by name if not in known applications
        return self._execute_command(app_name)
    
    def open_url(self, url):
        """Open a URL in the default browser.
        
        Args:
            url: URL to open
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.debug(f"Opening URL: {url}")
        
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            self.logger.error(f"Error opening URL: {e}")
            return False
    
    def _execute_command(self, command):
        """Execute a system command.
        
        Args:
            command: Command to execute
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.debug(f"Executing command: {command}")
        
        try:
            if self.os_type == 'win32':
                # Use start command on Windows to avoid blocking
                subprocess.Popen(f'start {command}', shell=True)
            else:
                # Use nohup on Unix-like systems to avoid blocking
                subprocess.Popen(command, shell=True, 
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
            return True
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return False
