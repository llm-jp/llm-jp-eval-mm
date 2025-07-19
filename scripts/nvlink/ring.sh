#!/bin/bash
# Slack notification script for SLURM jobs
# Usage: ./000_ring.sh <exit_code> <job_name>

# Get parameters
EXIT_CODE=${1:-0}
JOB_NAME=${2:-"Unknown Job"}

# Load environment variables from .env file
if [ -f "/home/silviase/augeo/.env" ]; then
    export $(grep -v '^#' /home/silviase/augeo/.env | xargs)
fi

# Check if webhook URL is set
if [ -z "$SLACK_WEBHOOK_URL" ]; then
    echo "Error: SLACK_WEBHOOK_URL not found in .env file"
    exit 1
fi

# Determine status based on exit code
if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="✅ SUCCESS"
    COLOR="good"
else
    STATUS="❌ FAILED (Exit code: $EXIT_CODE)"
    COLOR="danger"
fi

# Get SLURM job information
NODE=$(hostname)
JOB_ID=${SLURM_JOB_ID:-"N/A"}
CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# Send notification to Slack
curl -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-Type: application/json' \
    -d @- <<EOF
{
    "attachments": [{
        "color": "$COLOR",
        "title": "SLURM Job: $JOB_NAME",
        "fields": [
            {"title": "Status", "value": "$STATUS", "short": true},
            {"title": "Job ID", "value": "$JOB_ID", "short": true},
            {"title": "Node", "value": "$NODE", "short": true},
            {"title": "Completed", "value": "$CURRENT_TIME", "short": true}
        ],
        "footer": "Augeo Project",
        "ts": $(date +%s)
    }]
}
EOF

# Check if curl was successful
if [ $? -eq 0 ]; then
    echo "Slack notification sent successfully"
else
    echo "Failed to send Slack notification"
fi