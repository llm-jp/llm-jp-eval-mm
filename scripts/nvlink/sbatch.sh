#!/usr/bin/env bash
# How-to-use: bash scripts/nvlink/sbatch.sh MODEL_NAME [NUM_GPUS] [--tasks TASK1,TASK2,...]
# Example: bash scripts/nvlink/sbatch.sh "llava-hf/llava-1.5-7b-hf" 1 --tasks ai2d,mmmu
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# Initialize variables
SPECIFIED_TASKS=()
TASKS_MODE="all"

# === Argument Parsing ===
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 MODEL_NAME [NUM_GPUS] [--tasks TASK1,TASK2,...]"
    echo "       $0 --all [--tasks TASK1,TASK2,...]"
    echo ""
    echo "Options:"
    echo "  MODEL_NAME      The model to evaluate"
    echo "  NUM_GPUS       Number of GPUs to use (optional, auto-detected)"
    echo "  --all          Submit all models for evaluation"
    echo "  --tasks        Comma-separated list of tasks to run (optional, default: all tasks)"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Available tasks:"
    for task in "${task_list[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Examples:"
    echo "  # Single model with auto GPU detection, all tasks"
    echo "  $0 'llava-hf/llava-1.5-7b-hf'"
    echo ""
    echo "  # Single model with specific GPU count and specific tasks"
    echo "  $0 'OpenGVLab/InternVL3-78B' 4 --tasks ai2d,mmmu"
    echo ""
    echo "  # All models with specific tasks"
    echo "  $0 --all --tasks ai2d"
    echo ""
    echo "  # All models, all tasks"
    echo "  $0 --all"
    exit 0
fi

# Check if running in --all mode
if [ "$1" = "--all" ]; then
    ALL_MODE=true
    MODEL_NAME=""
    NUM_GPUS=""
    shift  # Remove --all from arguments
else
    ALL_MODE=false
    MODEL_NAME=$1
    NUM_GPUS=${2:-"auto"}
    shift  # Remove model name
    if [ "$NUM_GPUS" != "auto" ] && [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
        shift  # Remove NUM_GPUS if it's a number
    fi
fi

# Parse --tasks option
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS_MODE="specified"
            IFS=',' read -ra SPECIFIED_TASKS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which tasks to run
if [ "$TASKS_MODE" = "specified" ]; then
    # Validate specified tasks
    for task in "${SPECIFIED_TASKS[@]}"; do
        task_found=false
        for valid_task in "${task_list[@]}"; do
            if [ "$task" = "$valid_task" ]; then
                task_found=true
                break
            fi
        done
        if [ "$task_found" = false ]; then
            echo "❌ Invalid task: $task"
            echo "📋 Available tasks:"
            for valid_task in "${task_list[@]}"; do
                echo "   - $valid_task"
            done
            exit 1
        fi
    done
    tasks_to_run=("${SPECIFIED_TASKS[@]}")
else
    tasks_to_run=("${task_list[@]}")
fi

# Function to submit jobs for a single model
submit_single_model() {
    local model_name=$1
    local num_gpus=$2
    
    echo "🚀 Submitting jobs for model: $model_name"
    echo "🖥️  Number of GPUs: $num_gpus"
    echo "📋 Number of tasks: ${#tasks_to_run[@]}"
    if [ "$TASKS_MODE" = "specified" ]; then
        echo "📌 Running specific tasks: ${tasks_to_run[*]}"
    fi
    echo ""
    
    for task in "${tasks_to_run[@]}"; do
        mkdir -p "$REPO_PATH/outputs/$model_name/llm-jp-eval-mm/"
        echo "  📝 Submitting task: $task"
        
        # Use safe model name for job names and file paths (replace / with -)
        SAFE_MODEL_NAME=$(echo "$model_name" | sed 's/\//-/g')
        
        # Submit job directly with sbatch
        sbatch \
            --job-name="197_llm-jp-eval-mm_${SAFE_MODEL_NAME}_${task}" \
            --output="$REPO_PATH/outputs/$model_name/llm-jp-eval-mm/$task.out" \
            --error="$REPO_PATH/outputs/$model_name/llm-jp-eval-mm/$task.err" \
            --time=24:00:00 \
            --gres=gpu:$num_gpus \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem=64G \
            --wrap="bash $SCRIPT_PATH/eval.sh '$REPO_PATH' '$model_name' '$task' '$num_gpus'"
    done
    
    echo ""
    echo "✅ All tasks submitted successfully for $model_name!"
    echo "📊 Total jobs submitted: ${#tasks_to_run[@]}"
}

# Main execution
if [ "$ALL_MODE" = true ]; then
    # Submit all models
    echo "🚀 Starting batch submission for all models"
    echo "📊 Total models to evaluate: ${#model_gpu_map[@]}"
    if [ "$TASKS_MODE" = "specified" ]; then
        echo "📌 Running specific tasks: ${tasks_to_run[*]}"
    else
        echo "📋 Running all tasks"
    fi
    echo ""
    
    counter=0
    for model in "${!model_gpu_map[@]}"; do
        num_gpus=${model_gpu_map[$model]}
        counter=$((counter + 1))
        echo "[$counter/${#model_gpu_map[@]}] Processing $model with $num_gpus GPU(s)..."
        submit_single_model "$model" "$num_gpus"
        echo ""
        sleep 1  # Small delay between submissions
    done
    
    echo "✅ All models have been submitted!"
    echo "📊 Total models submitted: $counter"
else
    # Single model mode
    # Validate model name
    model_found=false
    for model in "${model_list[@]}"; do
        if [ "$MODEL_NAME" = "$model" ]; then
            model_found=true
            break
        fi
    done
    
    if [ "$model_found" = false ]; then
        echo "❌ Model '$MODEL_NAME' is not in the available model list."
        echo "📋 Available models:"
        for model in "${model_list[@]}"; do
            echo "   - $model"
        done
        exit 1
    fi
    
    # Auto-detect GPU count if needed
    if [ "$NUM_GPUS" = "auto" ]; then
        NUM_GPUS=${model_gpu_map[$MODEL_NAME]:-1}  # Default to 1 if not in map
        echo "🔍 Auto-detected GPU count: $NUM_GPUS"
    fi
    
    # Validate GPU count
    if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
        echo "❌ NUM_GPUS must be between 1 and 8"
        exit 1
    fi
    
    # Submit single model
    submit_single_model "$MODEL_NAME" "$NUM_GPUS"
fi