"""
Calendar Service - Manages calendar events and appointments.
"""

import os
import logging
import datetime
import json
import pickle
import tempfile
from pathlib import Path

# Try importing Google Calendar API
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False

class CalendarService:
    """Handles calendar operations using Google Calendar API."""
    
    def __init__(self):
        """Initialize the Calendar Service."""
        self.logger = logging.getLogger(__name__)
        
        # Google Calendar API settings
        self.scopes = ['https://www.googleapis.com/auth/calendar.readonly']
        self.api_service_name = 'calendar'
        self.api_version = 'v3'
        self.creds = None
        self.service = None
        
        # Path for token storage
        self.token_dir = Path(tempfile.gettempdir()) / 'jarvis'
        self.token_dir.mkdir(exist_ok=True)
        self.token_path = self.token_dir / 'token.pickle'
        
        # Check if Google Calendar API is available
        if not GOOGLE_CALENDAR_AVAILABLE:
            self.logger.warning("Google Calendar API not available. Install the required packages.")
        else:
            self._authenticate()
        
        self.logger.info("Calendar Service initialized")
    
    def _authenticate(self):
        """Authenticate with Google Calendar API."""
        if not GOOGLE_CALENDAR_AVAILABLE:
            return
        
        try:
            # Load credentials from token file
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)
            
            # Refresh or get new credentials
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    # Look for credentials.json
                    credentials_path = os.environ.get('GOOGLE_CREDENTIALS_PATH')
                    if not credentials_path:
                        self.logger.warning("Google Calendar credentials path not set in environment variables")
                        return
                    
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.scopes)
                    self.creds = flow.run_local_server(port=0)
                
                # Save credentials for future use
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
            
            # Build the service
            self.service = build(self.api_service_name, self.api_version, credentials=self.creds)
            self.logger.info("Successfully authenticated with Google Calendar")
            
        except Exception as e:
            self.logger.error(f"Error authenticating with Google Calendar: {e}")
    
    def get_events(self, days=7, max_results=10):
        """Get upcoming events from the calendar.
        
        Args:
            days: Number of days to look ahead
            max_results: Maximum number of events to return
            
        Returns:
            List of event dictionaries or None if error
        """
        if not GOOGLE_CALENDAR_AVAILABLE or not self.service:
            self.logger.warning("Google Calendar service not available")
            return None
        
        try:
            # Calculate time range
            now = datetime.datetime.utcnow()
            end_time = now + datetime.timedelta(days=days)
            
            # Format timestamps for API
            now_str = now.isoformat() + 'Z'  # 'Z' indicates UTC time
            end_str = end_time.isoformat() + 'Z'
            
            # Query the API
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now_str,
                timeMax=end_str,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Format events into a list of dictionaries
            formatted_events = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                
                if 'T' in start:  # DateTime format
                    start_time = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:  # Date format
                    start_time = datetime.datetime.strptime(start, '%Y-%m-%d')
                
                event_info = {
                    'summary': event.get('summary', 'Untitled Event'),
                    'start_time': start_time,
                    'location': event.get('location', ''),
                    'description': event.get('description', ''),
                    'link': event.get('htmlLink', '')
                }
                
                formatted_events.append(event_info)
            
            return formatted_events
            
        except Exception as e:
            self.logger.error(f"Error retrieving events from Google Calendar: {e}")
            return None
    
    def get_events_for_today(self):
        """Get events scheduled for today.
        
        Returns:
            List of event summaries or empty list if none/error
        """
        events = self.get_events(days=1)
        if not events:
            return []
        
        # Filter events for today
        today = datetime.datetime.now().date()
        today_events = [event for event in events 
                       if event['start_time'].date() == today]
        
        # Return just the summaries
        return [event['summary'] for event in today_events]
    
    def get_events_for_tomorrow(self):
        """Get events scheduled for tomorrow.
        
        Returns:
            List of event summaries or empty list if none/error
        """
        events = self.get_events(days=2)
        if not events:
            return []
        
        # Filter events for tomorrow
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
        tomorrow_events = [event for event in events 
                          if event['start_time'].date() == tomorrow]
        
        # Return just the summaries
        return [event['summary'] for event in tomorrow_events]
    
    def get_formatted_events(self, days=7):
        """Get a formatted string with upcoming events.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            Formatted string with events
        """
        events = self.get_events(days=days)
        
        if not events:
            return "No upcoming events found or calendar service unavailable."
        
        result = f"Upcoming events for the next {days} days:\n\n"
        
        # Group events by day
        events_by_day = {}
        for event in events:
            day = event['start_time'].date()
            if day not in events_by_day:
                events_by_day[day] = []
            events_by_day[day].append(event)
        
        # Format each day's events
        for day, day_events in sorted(events_by_day.items()):
            result += f"{day.strftime('%A, %B %d')}:\n"
            
            for event in day_events:
                time_str = event['start_time'].strftime('%H:%M') if isinstance(event['start_time'], datetime.datetime) else 'All day'
                result += f"  {time_str} - {event['summary']}"
                
                if event['location']:
                    result += f" at {event['location']}"
                    
                result += "\n"
            
            result += "\n"
        
        return result
