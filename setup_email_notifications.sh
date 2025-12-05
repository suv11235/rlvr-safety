#!/bin/bash
# Quick setup script for email notifications

echo "=================================================="
echo "Email Notification Setup"
echo "=================================================="
echo ""
echo "Setting up email for: majumder.suvajit95@gmail.com"
echo ""
echo "You need a Gmail App Password to continue."
echo ""
echo "📋 Steps to get your App Password:"
echo "   1. Go to: https://myaccount.google.com/apppasswords"
echo "   2. Sign in with your Google account"
echo "   3. Create an app password for 'Training Notifications'"
echo "   4. Copy the 16-character password"
echo ""
read -p "Do you have your Gmail App Password ready? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "❌ Setup cancelled. Get your App Password first, then run this script again."
    echo ""
    exit 1
fi

echo ""
read -sp "Enter your Gmail App Password (16 chars, no spaces): " APP_PASSWORD
echo ""
echo ""

if [ -z "$APP_PASSWORD" ]; then
    echo "❌ No password entered. Setup cancelled."
    exit 1
fi

# Create credentials file
CRED_FILE="$HOME/email_credentials.sh"

cat > "$CRED_FILE" << EOF
# Email notification credentials
# Created: $(date)
export NOTIFICATION_EMAIL="majumder.suvajit95@gmail.com"
export GMAIL_ADDRESS="majumder.suvajit95@gmail.com"
export GMAIL_APP_PASSWORD="$APP_PASSWORD"
EOF

# Secure the file
chmod 600 "$CRED_FILE"

echo "✅ Credentials saved to: $CRED_FILE"
echo "   (File is secured with chmod 600)"
echo ""

# Load credentials
source "$CRED_FILE"

# Test email
echo "📧 Testing email notification..."
echo ""

cd /lambda/nfs/mars-2xh100/suvajit/rlvr-safety

if bash scripts/notify.sh "Email notifications setup complete! You will receive alerts when training starts." "info"; then
    echo ""
    echo "=================================================="
    echo "✅ Email Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Check your email at: majumder.suvajit95@gmail.com"
    echo ""
    echo "Next steps:"
    echo "1. Verify you received the test email"
    echo "2. To make credentials permanent, run:"
    echo "   echo 'source ~/email_credentials.sh' >> ~/.bashrc"
    echo ""
    echo "To restart scheduler with email notifications:"
    echo "   pkill -f schedule_defence.sh"
    echo "   source ~/email_credentials.sh"
    echo "   cd /lambda/nfs/mars-2xh100/suvajit/rlvr-safety"
    echo "   nohup ./schedule_defence.sh > defence_scheduler.log 2>&1 &"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "⚠️  Email Test Failed"
    echo "=================================================="
    echo ""
    echo "The credentials were saved, but the test email failed."
    echo "This might be due to:"
    echo "  - Incorrect App Password"
    echo "  - 2FA not enabled on your Google account"
    echo "  - Network issues"
    echo ""
    echo "Check EMAIL_SETUP.md for troubleshooting."
    echo ""
fi
