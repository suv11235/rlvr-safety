# Email Notification Setup Guide

## Quick Setup for Gmail

### Step 1: Get a Gmail App Password

Since you're using **majumder.suvajit95@gmail.com**, you need to create an App Password:

1. **Go to:** https://myaccount.google.com/apppasswords
2. **Sign in** with your Google account
3. **Create App Password:**
   - App name: "Training Notifications" (or any name)
   - Click "Create"
4. **Copy the 16-character password** (looks like: `xxxx xxxx xxxx xxxx`)

### Step 2: Set Environment Variables

Create a file with your credentials (never commit this to git!):

```bash
# Create credentials file
cat > ~/email_credentials.sh << 'EOF'
# Email notification credentials
export NOTIFICATION_EMAIL="majumder.suvajit95@gmail.com"
export GMAIL_ADDRESS="majumder.suvajit95@gmail.com"
export GMAIL_APP_PASSWORD="your-16-char-app-password-here"
EOF

# Secure the file
chmod 600 ~/email_credentials.sh
```

**Replace** `your-16-char-app-password-here` with the App Password from Step 1.

### Step 3: Restart the Scheduler with Email Notifications

```bash
# Stop current scheduler
pkill -f schedule_defence.sh

# Load email credentials
source ~/email_credentials.sh

# Restart scheduler with email notifications
cd /lambda/nfs/mars-2xh100/suvajit/rlvr-safety
nohup ./schedule_defence.sh > defence_scheduler.log 2>&1 &
```

### Step 4: Test Email Notifications

```bash
# Load credentials
source ~/email_credentials.sh

# Send test email
bash scripts/notify.sh "Test email notification - System is working!" "info"
```

You should receive an email at **majumder.suvajit95@gmail.com** within a few seconds.

## Alternative: Using Other Email Providers

### For Other SMTP Servers

If you don't want to use Gmail, set these variables instead:

```bash
export NOTIFICATION_EMAIL="majumder.suvajit95@gmail.com"
export SMTP_SERVER="smtp.your-provider.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-username"
export SMTP_PASSWORD="your-password"
export SMTP_USE_TLS="true"
```

Common SMTP configurations:

**Outlook/Hotmail:**
```bash
export SMTP_SERVER="smtp-mail.outlook.com"
export SMTP_PORT="587"
export SMTP_USE_TLS="true"
```

**Yahoo:**
```bash
export SMTP_SERVER="smtp.mail.yahoo.com"
export SMTP_PORT="587"
export SMTP_USE_TLS="true"
```

**Custom SMTP:**
```bash
export SMTP_SERVER="mail.yourdomain.com"
export SMTP_PORT="465"  # or 587
export SMTP_USE_TLS="true"
```

## Troubleshooting

### "Authentication failed" Error

**Gmail:** Make sure you're using an App Password, not your regular Gmail password.
- Go to: https://myaccount.google.com/apppasswords
- 2-Factor Authentication must be enabled first

### "Connection refused" Error

Check your SMTP server and port are correct:
```bash
echo $SMTP_SERVER
echo $SMTP_PORT
```

### Test SMTP Connection

```bash
python3 -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
print('✓ SMTP connection successful')
server.quit()
"
```

## Email Notification Events

You'll receive emails for:
- 🚀 **Training Start** - When defense training begins
- ✅ **Training Complete** - When training finishes
- ❌ **Errors** - If training encounters errors

## Security Notes

1. **Never commit** `~/email_credentials.sh` to git
2. The credentials file is secured with `chmod 600` (only you can read it)
3. App Passwords are safer than your main Gmail password
4. You can revoke App Passwords anytime at https://myaccount.google.com/apppasswords

## Persistent Setup

To make email credentials available every time you log in:

```bash
# Add to your .bashrc
echo "source ~/email_credentials.sh" >> ~/.bashrc
```

Now credentials will be loaded automatically in new terminal sessions.
