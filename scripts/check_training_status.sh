#!/bin/bash
# Quick status check for scheduled and running training

echo "=================================================="
echo "Training Status Report"
echo "=================================================="
echo "Generated at: $(date)"
echo ""

# Check if scheduler is running
echo "📅 Scheduled Training:"
if pgrep -f schedule_defence.sh > /dev/null; then
    echo "   ✓ Defense training scheduler is RUNNING"
    SCHEDULER_PID=$(pgrep -f schedule_defence.sh)
    echo "   PID: $SCHEDULER_PID"

    # Calculate remaining time
    if [ -f defence_scheduler.log ]; then
        SCHEDULED_TIME=$(grep "Scheduled start:" defence_scheduler.log | head -1 | awk '{print $3, $4}')
        if [ -n "$SCHEDULED_TIME" ]; then
            echo "   Will start at: $SCHEDULED_TIME"

            CURRENT_EPOCH=$(date +%s)
            SCHEDULED_EPOCH=$(date -d "$SCHEDULED_TIME" +%s 2>/dev/null)
            if [ $? -eq 0 ]; then
                REMAINING=$((SCHEDULED_EPOCH - CURRENT_EPOCH))
                if [ $REMAINING -gt 0 ]; then
                    HOURS=$((REMAINING / 3600))
                    MINUTES=$(( (REMAINING % 3600) / 60 ))
                    echo "   Time remaining: ${HOURS}h ${MINUTES}m"
                else
                    echo "   Status: Should have started already, check if training is running"
                fi
            fi
        fi
    fi
else
    echo "   ✗ No scheduled training found"
fi

echo ""

# Check if attack training is running
echo "⚔️  Attack Training:"
if pgrep -f "main_ppo.*attack" > /dev/null; then
    ATTACK_PID=$(pgrep -f "main_ppo.*attack" | head -1)
    echo "   ✓ Attack training is RUNNING (PID: $ATTACK_PID)"

    # Get latest checkpoint
    LATEST_ATTACK_CKPT=$(ls -dt ./outputs/harmfulrl*/global_step_* 2>/dev/null | head -1)
    if [ -n "$LATEST_ATTACK_CKPT" ]; then
        STEP=$(basename "$LATEST_ATTACK_CKPT" | sed 's/global_step_//')
        echo "   Latest checkpoint: Step $STEP"
        echo "   Saved at: $(stat -c %y "$LATEST_ATTACK_CKPT" | cut -d'.' -f1)"
    fi
else
    echo "   ✗ Not running"
    # Check for completed checkpoints
    LATEST_ATTACK_CKPT=$(ls -dt ./outputs/harmfulrl*/global_step_* 2>/dev/null | head -1)
    if [ -n "$LATEST_ATTACK_CKPT" ]; then
        STEP=$(basename "$LATEST_ATTACK_CKPT" | sed 's/global_step_//')
        echo "   Last checkpoint: Step $STEP ($(stat -c %y "$LATEST_ATTACK_CKPT" | cut -d'.' -f1))"
    fi
fi

echo ""

# Check if defense training is running
echo "🛡️  Defense Training:"
if pgrep -f "main_ppo.*defence\|main_ppo.*defense\|tokenbuncher" > /dev/null; then
    DEFENSE_PID=$(pgrep -f "main_ppo.*defence\|main_ppo.*defense\|tokenbuncher" | head -1)
    echo "   ✓ Defense training is RUNNING (PID: $DEFENSE_PID)"

    # Get latest checkpoint
    LATEST_DEFENSE_CKPT=$(ls -dt ./outputs/*defence*/global_step_* ./outputs/*defense*/global_step_* ./outputs/*tokenbuncher*/global_step_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DEFENSE_CKPT" ]; then
        STEP=$(basename "$LATEST_DEFENSE_CKPT" | sed 's/global_step_//')
        echo "   Latest checkpoint: Step $STEP"
        echo "   Saved at: $(stat -c %y "$LATEST_DEFENSE_CKPT" | cut -d'.' -f1)"
    fi
else
    echo "   ✗ Not running"
    # Check for completed checkpoints
    LATEST_DEFENSE_CKPT=$(ls -dt ./outputs/*defence*/global_step_* ./outputs/*defense*/global_step_* ./outputs/*tokenbuncher*/global_step_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DEFENSE_CKPT" ]; then
        STEP=$(basename "$LATEST_DEFENSE_CKPT" | sed 's/global_step_//')
        echo "   Last checkpoint: Step $STEP ($(stat -c %y "$LATEST_DEFENSE_CKPT" | cut -d'.' -f1))"
    fi
fi

echo ""

# Show recent notifications
echo "🔔 Recent Notifications:"
if [ -f "./LATEST_NOTIFICATION.txt" ]; then
    echo "   Latest notification:"
    sed 's/^/   /' ./LATEST_NOTIFICATION.txt | head -10
else
    echo "   No notifications yet"
fi

echo ""

# Show notification markers
MARKERS=$(ls -t TRAINING_*.marker 2>/dev/null | head -3)
if [ -n "$MARKERS" ]; then
    echo "📍 Recent Events:"
    for marker in $MARKERS; do
        echo "   - $marker"
        sed 's/^/     /' "$marker"
    done
else
    echo "📍 No event markers found"
fi

echo ""
echo "=================================================="
echo "Tip: Run 'bash scripts/watch_notifications.sh' to monitor in real-time"
echo "=================================================="
