"""
Email Service - Handles sending and reading emails.
"""

import os
import logging
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

class EmailService:
    """Handles email operations."""
    
    def __init__(self):
        """Initialize the Email Service."""
        self.logger = logging.getLogger(__name__)
        
        # Email credentials from environment variables
        self.email_address = os.environ.get("EMAIL_ADDRESS", "")
        self.email_password = os.environ.get("EMAIL_PASSWORD", "")
        
        # Default SMTP and IMAP settings
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.imap_server = os.environ.get("IMAP_SERVER", "imap.gmail.com")
        self.imap_port = int(os.environ.get("IMAP_PORT", "993"))
        
        if not self.email_address or not self.email_password:
            self.logger.warning("Email credentials not found in environment variables")
        
        self.logger.info("Email Service initialized")
    
    def send_email(self, recipient, subject, body, html_body=None):
        """Send an email.
        
        Args:
            recipient: Email address of the recipient
            subject: Subject of the email
            body: Text body of the email
            html_body: Optional HTML body of the email
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_address or not self.email_password:
            self.logger.error("Email credentials not available")
            return False
        
        self.logger.debug(f"Sending email to {recipient}")
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Attach text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Attach HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    def read_emails(self, folder='inbox', limit=5, unread_only=False):
        """Read emails from the specified folder.
        
        Args:
            folder: Email folder to read from
            limit: Maximum number of emails to read
            unread_only: Whether to read only unread emails
            
        Returns:
            List of email dictionaries or empty list if error
        """
        if not self.email_address or not self.email_password:
            self.logger.error("Email credentials not available")
            return []
        
        self.logger.debug(f"Reading emails from {folder}")
        
        try:
            # Connect to IMAP server
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.email_address, self.email_password)
            mail.select(folder)
            
            # Search for emails
            search_criteria = '(UNSEEN)' if unread_only else 'ALL'
            status, data = mail.search(None, search_criteria)
            
            if status != 'OK':
                self.logger.error(f"Error searching emails: {status}")
                return []
            
            # Get email IDs
            email_ids = data[0].split()
            if not email_ids:
                self.logger.info(f"No {'unread ' if unread_only else ''}emails found in {folder}")
                return []
            
            # Limit the number of emails to process
            email_ids = email_ids[-limit:] if limit else email_ids
            
            # Process emails
            emails = []
            for e_id in reversed(email_ids):
                status, data = mail.fetch(e_id, '(RFC822)')
                if status != 'OK':
                    continue
                
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Extract email parts
                subject = self._decode_header(msg['Subject'])
                from_addr = self._decode_header(msg['From'])
                date_str = msg['Date']
                
                # Try to parse the date
                try:
                    date = email.utils.parsedate_to_datetime(date_str)
                except:
                    date = None
                
                # Get email body
                body = self._get_email_body(msg)
                
                # Create email object
                email_obj = {
                    'id': e_id.decode(),
                    'subject': subject,
                    'from': from_addr,
                    'date': date,
                    'body': body
                }
                
                emails.append(email_obj)
            
            mail.close()
            mail.logout()
            
            return emails
            
        except Exception as e:
            self.logger.error(f"Error reading emails: {e}")
            return []
    
    def _decode_header(self, header):
        """Decode email header.
        
        Args:
            header: Email header to decode
            
        Returns:
            Decoded header string
        """
        if not header:
            return ""
        
        try:
            decoded_header = email.header.decode_header(header)
            if decoded_header:
                # Handle multi-part headers
                parts = []
                for decoded_text, charset in decoded_header:
                    if isinstance(decoded_text, bytes):
                        try:
                            if charset:
                                parts.append(decoded_text.decode(charset or 'utf-8', errors='replace'))
                            else:
                                parts.append(decoded_text.decode('utf-8', errors='replace'))
                        except:
                            parts.append(decoded_text.decode('utf-8', errors='replace'))
                    else:
                        parts.append(decoded_text)
                
                return ''.join(parts)
            else:
                return header
        except:
            return header
    
    def _get_email_body(self, msg):
        """Extract the body from an email message.
        
        Args:
            msg: Email message
            
        Returns:
            Body text
        """
        body = ""
        
        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Try to get the body
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        pass
                        
                elif content_type == "text/html" and not body:
                    try:
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
        else:
            # Handle non-multipart messages
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                body = msg.get_payload()
        
        return body
    
    def get_formatted_emails(self, folder='inbox', limit=3, unread_only=True):
        """Get a formatted string with emails.
        
        Args:
            folder: Email folder to read from
            limit: Maximum number of emails to read
            unread_only: Whether to read only unread emails
            
        Returns:
            Formatted string with emails
        """
        emails = self.read_emails(folder, limit, unread_only)
        
        if not emails:
            return f"No {'unread ' if unread_only else ''}emails found in {folder}."
        
        result = f"{'Unread ' if unread_only else ''}Emails in {folder}:\n\n"
        
        for email_obj in emails:
            # Format the date
            date_str = "Unknown date"
            if email_obj['date']:
                date_str = email_obj['date'].strftime('%B %d, %Y at %I:%M %p')
            
            result += f"From: {email_obj['from']}\n"
            result += f"Subject: {email_obj['subject']}\n"
            result += f"Date: {date_str}\n"
            
            # Truncate body if too long
            body = email_obj['body']
            if len(body) > 200:
                body = body[:200] + "..."
            
            result += f"Body: {body}\n\n"
        
        return result
