#!/usr/bin/env bash
# Monitor SLURM jobs for llm-jp-eval-mm
set -euo pipefail

# Configuration
ROOT_DIR="/home/silviase/"
REPO_PATH="$ROOT_DIR/llm-jp-eval-mm"
OUTPUT_DIR="$REPO_PATH/outputs"
REFRESH_INTERVAL=${1:-5}  # Default 5 seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to get job status with color
get_status_color() {
    case $1 in
        "RUNNING") echo -e "${GREEN}RUNNING${NC}" ;;
        "PENDING") echo -e "${YELLOW}PENDING${NC}" ;;
        "COMPLETED") echo -e "${BLUE}COMPLETED${NC}" ;;
        "FAILED") echo -e "${RED}FAILED${NC}" ;;
        "CANCELLED") echo -e "${PURPLE}CANCELLED${NC}" ;;
        *) echo -e "$1" ;;
    esac
}

# Function to format time
format_time() {
    local time=$1
    if [[ $time == *-* ]]; then
        # Format: days-HH:MM:SS
        echo "$time"
    else
        # Format: HH:MM:SS
        echo "$time"
    fi
}

# Main monitoring loop
while true; do
    # Build entire output in a variable first
    output=""
    output+="\033[H\033[2J"  # Clear screen
    output+="${CYAN}=== LLM-JP-EVAL-MM Job Monitor ===${NC}\n"
    output+="$(date '+%Y-%m-%d %H:%M:%S') | Refresh: ${REFRESH_INTERVAL}s\n"
    output+="\n"

    # Get all jobs for current user
    jobs_output=$(squeue -u $USER --format="%.18i %.40j %.8T %.10M %.6D %.4C %.Z" --noheader || true)
    
    if [ -z "$jobs_output" ]; then
        output+="${YELLOW}No active jobs found${NC}\n"
    else
        # Header
        output+=$(printf "%-18s %-40s %-12s %-10s %-6s %-6s\n" \
            "JOB ID" "NAME" "STATUS" "TIME" "NODES" "CPUS")
        output+="$(printf '%.0s-' {1..100})\n"
        
        # Process each job
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                job_id=$(echo "$line" | awk '{print $1}')
                job_name=$(echo "$line" | awk '{print $2}' | cut -c1-40)
                status=$(echo "$line" | awk '{print $3}')
                time=$(echo "$line" | awk '{print $4}')
                nodes=$(echo "$line" | awk '{print $5}')
                cpus=$(echo "$line" | awk '{print $6}')
                
                # Color status
                status_colored=$(get_status_color "$status")
                
                # Format time
                time_formatted=$(format_time "$time")
                
                output+=$(printf "%-18s %-40s %-22s %-10s %-6s %-6s\n" \
                    "$job_id" "$job_name" "$status_colored" "$time_formatted" "$nodes" "$cpus")
            fi
        done <<< "$jobs_output"
    fi
    
    output+="\n"
    output+="$(printf '%.0s-' {1..100})\n"
    
    # Count jobs by status
    output+="${CYAN}Summary:${NC}\n"
    if [ -z "$jobs_output" ]; then
        total_jobs=0
        running_jobs=0
        pending_jobs=0
    else
        total_jobs=$(echo "$jobs_output" | grep -c . || echo "0")
        running_jobs=$(echo "$jobs_output" | grep -c "RUNNING" || echo "0")
        pending_jobs=$(echo "$jobs_output" | grep -c "PENDING" || echo "0")
    fi
    
    output+="Total: $total_jobs | Running: ${GREEN}$running_jobs${NC} | Pending: ${YELLOW}$pending_jobs${NC}\n"
    
    # Show recent completions from sacct
    output+="\n"
    output+="${CYAN}Recent Completions (last hour):${NC}\n"
    recent_jobs=$(sacct -u $USER -S $(date -d '1 hour ago' '+%Y-%m-%d') \
        --format="JobID%20,JobName%40,State%15,Elapsed%12,ExitCode" \
        --noheader 2>/dev/null | grep -E "(COMPLETED|FAILED|CANCELLED)" | head -10 || true)
    
    if [ -z "$recent_jobs" ]; then
        output+="No recent completions\n"
    else
        output+=$(printf "%-20s %-40s %-15s %-12s %-8s\n" \
            "JOB ID" "NAME" "STATE" "ELAPSED" "EXIT")
        output+="$(printf '%.0s-' {1..100})\n"
        
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                job_id=$(echo "$line" | awk '{print $1}')
                job_name=$(echo "$line" | awk '{print $2}' | cut -c1-40)
                state=$(echo "$line" | awk '{print $3}')
                elapsed=$(echo "$line" | awk '{print $4}')
                exit_code=$(echo "$line" | awk '{print $5}')
                
                # Color state
                state_colored=$(get_status_color "$state")
                
                output+=$(printf "%-20s %-40s %-25s %-12s %-8s\n" \
                    "$job_id" "$job_name" "$state_colored" "$elapsed" "$exit_code")
            fi
        done <<< "$recent_jobs"
    fi
    
    output+="\n"
    output+="Press Ctrl+C to exit\n"
    
    # Print all output at once
    echo -e "$output"
    
    sleep $REFRESH_INTERVAL
done