#!/usr/bin/env bash
# How-to-use: bash scripts/nvlink/sbatch.sh MODEL_NAME [NUM_GPUS]
# Example: bash scripts/nvlink/sbatch.sh "llava-hf/llava-1.5-7b-hf" 1
set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# === Argument Parsing ===
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 MODEL_NAME [NUM_GPUS]"
    echo "       $0 --all"
    echo ""
    echo "Options:"
    echo "  MODEL_NAME      The model to evaluate"
    echo "  NUM_GPUS       Number of GPUs to use (optional, auto-detected)"
    echo "  --all          Submit all models for evaluation"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single model with auto GPU detection"
    echo "  $0 'llava-hf/llava-1.5-7b-hf'"
    echo ""
    echo "  # Single model with specific GPU count"
    echo "  $0 'OpenGVLab/InternVL3-78B' 4"
    echo ""
    echo "  # All models"
    echo "  $0 --all"
    exit 0
fi

# Check if running in --all mode
if [ "$1" = "--all" ]; then
    ALL_MODE=true
    MODEL_NAME=""
    NUM_GPUS=""
else
    ALL_MODE=false
    MODEL_NAME=$1
    NUM_GPUS=${2:-"auto"}
fi

# Function to submit jobs for a single model
submit_single_model() {
    local model_name=$1
    local num_gpus=$2
    
    echo "🚀 Submitting jobs for model: $model_name"
    echo "🖥️  Number of GPUs: $num_gpus"
    echo "📋 Number of tasks: ${#task_list[@]}"
    echo ""
    
    for task in "${task_list[@]}"; do
        mkdir -p "$REPO_PATH/outputs/$model_name/llm-jp-eval-mm/"
        echo "  📝 Submitting task: $task"
        
        # Use safe model name for job names and file paths (replace / with -)
        SAFE_MODEL_NAME=$(echo "$model_name" | sed 's/\//-/g')
        
        # Submit job directly with sbatch
        sbatch \
            --job-name="llm-jp-eval-mm_${SAFE_MODEL_NAME}_${task}" \
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
    echo "📊 Total jobs submitted: ${#task_list[@]}"
}

# Main execution
if [ "$ALL_MODE" = true ]; then
    # Submit all models
    echo "🚀 Starting batch submission for all models"
    echo "📊 Total models to evaluate: ${#model_gpu_map[@]}"
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