#!/bin/bash
# Watch for training notifications in real-time

echo "=================================================="
echo "Training Notification Monitor"
echo "=================================================="
echo "Watching for notifications..."
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

# Show latest notification if exists
if [ -f "./LATEST_NOTIFICATION.txt" ]; then
    echo "Most recent notification:"
    cat ./LATEST_NOTIFICATION.txt
    echo ""
fi

# Watch for changes in notification log and marker files
echo "Monitoring for new notifications..."
echo ""

while true; do
    # Check for new marker files
    NEW_MARKERS=$(find . -maxdepth 1 -name "TRAINING_*.marker" -mmin -1 2>/dev/null)

    if [ -n "$NEW_MARKERS" ]; then
        echo ""
        echo "🔔 NEW NOTIFICATION DETECTED!"
        echo "=================================================="
        cat ./LATEST_NOTIFICATION.txt 2>/dev/null
        echo "=================================================="
        echo ""
    fi

    sleep 30
done
