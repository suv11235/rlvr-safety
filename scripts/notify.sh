#!/bin/bash
# Notification script for training events

MESSAGE="$1"
EVENT_TYPE="${2:-info}"  # info, start, complete, error
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
NOTIFICATION_LOG="./notifications.log"

# Function to log notification
log_notification() {
    echo "[$TIMESTAMP] [$EVENT_TYPE] $MESSAGE" >> "$NOTIFICATION_LOG"
}

# Function to create marker file
create_marker() {
    local marker_file="./TRAINING_${EVENT_TYPE}_$(date +%s).marker"
    echo "[$TIMESTAMP] $MESSAGE" > "$marker_file"
    echo "$marker_file"
}

# Function to send email (using Python SMTP)
send_email() {
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        python3 "$(dirname "$0")/send_email_notification.py" \
            "$NOTIFICATION_EMAIL" \
            "🤖 Training Notification: $EVENT_TYPE" \
            "$MESSAGE" 2>&1
    fi
}

# Function to send Slack notification (if webhook is configured)
send_slack() {
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"🤖 *Training Notification*\n\`\`\`\n[$EVENT_TYPE] $MESSAGE\n\`\`\`\"}" \
            2>/dev/null
    fi
}

# Function to send Discord notification (if webhook is configured)
send_discord() {
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -X POST "$DISCORD_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"content\":\"🤖 **Training Notification**\n\`\`\`\n[$EVENT_TYPE] $MESSAGE\n\`\`\`\"}" \
            2>/dev/null
    fi
}

# Main notification logic
case $EVENT_TYPE in
    start)
        COLOR=$GREEN
        ICON="🚀"
        ;;
    complete)
        COLOR=$BLUE
        ICON="✅"
        ;;
    error)
        COLOR=$RED
        ICON="❌"
        ;;
    *)
        COLOR=$YELLOW
        ICON="ℹ️"
        ;;
esac

# Print to console with color
echo -e "${COLOR}${ICON} [$EVENT_TYPE] $MESSAGE${NC}"

# Log to file
log_notification

# Create marker file
marker_file=$(create_marker)
echo "Marker created: $marker_file"

# Send email if configured
if [ -n "$NOTIFICATION_EMAIL" ]; then
    send_email
    echo "Email notification sent to $NOTIFICATION_EMAIL"
fi

# Send Slack if configured
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    send_slack
    echo "Slack notification sent"
fi

# Send Discord if configured
if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    send_discord
    echo "Discord notification sent"
fi

# Write to a prominent notification file
NOTIFICATION_FILE="./LATEST_NOTIFICATION.txt"
cat > "$NOTIFICATION_FILE" << EOF
========================================
TRAINING NOTIFICATION
========================================
Time:    $TIMESTAMP
Event:   $EVENT_TYPE
Message: $MESSAGE
========================================

Check notifications.log for history.
EOF

echo "Notification logged to: $NOTIFICATION_FILE"
