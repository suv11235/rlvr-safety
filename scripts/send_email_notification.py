#!/usr/bin/env python3
"""
Send email notifications for training events.
Uses Gmail SMTP or other SMTP providers.
"""

import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

def send_email(to_email, subject, message, from_email=None, smtp_config=None):
    """
    Send email notification.

    For Gmail, you need an App Password:
    1. Go to https://myaccount.google.com/apppasswords
    2. Create an app password for "Mail"
    3. Set GMAIL_APP_PASSWORD environment variable
    """

    # Default SMTP configuration (Gmail)
    if smtp_config is None:
        smtp_config = {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            'username': os.getenv('SMTP_USERNAME') or os.getenv('GMAIL_ADDRESS'),
            'password': os.getenv('SMTP_PASSWORD') or os.getenv('GMAIL_APP_PASSWORD'),
        }

    # Use environment variable for from_email if not specified
    if from_email is None:
        from_email = smtp_config['username']

    if not smtp_config['username'] or not smtp_config['password']:
        print("❌ Error: SMTP credentials not configured!")
        print("   For Gmail, set these environment variables:")
        print("   - GMAIL_ADDRESS='your.email@gmail.com'")
        print("   - GMAIL_APP_PASSWORD='your-app-password'")
        print("   Get app password from: https://myaccount.google.com/apppasswords")
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add timestamp to message body
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"Timestamp: {timestamp}\n\n{message}"

        msg.attach(MIMEText(full_message, 'plain'))

        # Connect to SMTP server and send
        print(f"📧 Connecting to {smtp_config['server']}:{smtp_config['port']}...")

        server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])

        if smtp_config['use_tls']:
            server.starttls()

        print(f"🔐 Authenticating as {smtp_config['username']}...")
        server.login(smtp_config['username'], smtp_config['password'])

        print(f"📤 Sending email to {to_email}...")
        server.send_message(msg)
        server.quit()

        print(f"✅ Email sent successfully to {to_email}!")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python send_email_notification.py <to_email> <subject> <message>")
        sys.exit(1)

    to_email = sys.argv[1]
    subject = sys.argv[2]
    message = sys.argv[3] if len(sys.argv) > 3 else ""

    success = send_email(to_email, subject, message)
    sys.exit(0 if success else 1)
