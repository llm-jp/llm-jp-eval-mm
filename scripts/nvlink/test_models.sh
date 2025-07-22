#!/bin/bash
# Unified model testing script
# Usage:
#   Test all models:     bash scripts/nvlink/test_models.sh
#   Test single model:   bash scripts/nvlink/test_models.sh --model "MODEL_ID"
#   Submit via sbatch:   bash scripts/nvlink/test_models.sh --submit [--model "MODEL_ID"] [--gpus N]
#   Interactive mode:    bash scripts/nvlink/test_models.sh --interactive [--model "MODEL_ID"]

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# Default values
MODE="direct"  # direct, submit, interactive
MODEL_ID=""
NUM_GPUS="auto"
TIME_LIMIT="1:00:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --submit|-s)
            MODE="submit"
            shift
            ;;
        --interactive|-i)
            MODE="interactive"
            shift
            ;;
        --model|-m)
            MODEL_ID="$2"
            shift 2
            ;;
        --gpus|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --time|-t)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --submit, -s         Submit job via sbatch"
            echo "  --interactive, -i    Run in interactive mode"
            echo "  --model, -m MODEL    Test specific model (default: all models)"
            echo "  --gpus, -g N        Number of GPUs (default: auto-detect)"
            echo "  --time, -t TIME     Time limit (default: 1:00:00)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Test all models directly"
            echo "  $0"
            echo ""
            echo "  # Test single model with auto GPU detection"
            echo "  $0 --model 'OpenGVLab/InternVL3-8B'"
            echo ""
            echo "  # Submit single model test with 4 GPUs"
            echo "  $0 --submit --model 'OpenGVLab/InternVL3-78B' --gpus 4"
            echo ""
            echo "  # Submit all models test (6 hours)"
            echo "  $0 --submit --time 6:00:00"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to test a single model
test_single_model() {
    local model_name=$1
    local model_group=${MODEL_GROUP_MAP[$model_name]:-"normal"}
    
    echo "Testing model: $model_name (env: $model_group)"
    
    cd $REPO_PATH
    source .uv/$model_group-env/bin/activate
    python examples/test_model.py --model_id "$model_name"
    
    echo "Completed testing: $model_name"
    echo "---"
}

# Function to test all models
test_all_models() {
    cd $REPO_PATH
    
    # Load .env file
    load_env_file
    
    # Test all models
    for model_name in "${!MODEL_GROUP_MAP[@]}"; do
        test_single_model "$model_name"
    done
    
    echo "All model tests completed."
}

# Main execution logic
case $MODE in
    direct)
        # Direct execution
        export PATH="$HOME/.local/bin:$PATH"
        
        if [ -n "$MODEL_ID" ]; then
            # Test single model
            load_env_file
            test_single_model "$MODEL_ID"
        else
            # Test all models
            test_all_models
        fi
        ;;
        
    submit)
        # Submit via sbatch
        if [ -n "$MODEL_ID" ]; then
            # Single model submission
            if [ "$NUM_GPUS" = "auto" ]; then
                NUM_GPUS=${model_gpu_map[$MODEL_ID]:-1}
                echo "🔍 Auto-detected GPU count: $NUM_GPUS"
            fi
            
            SAFE_MODEL_NAME=$(echo "$MODEL_ID" | sed 's/\//-/g')
            JOB_NAME="test-${SAFE_MODEL_NAME}"
            
            echo "📝 Submitting test job for: $MODEL_ID"
            echo "🖥️  Number of GPUs: $NUM_GPUS"
            echo "⏱️  Time limit: $TIME_LIMIT"
            
            sbatch \
                --job-name="$JOB_NAME" \
                --time="$TIME_LIMIT" \
                --gres=gpu:$NUM_GPUS \
                --ntasks=1 \
                --cpus-per-task=8 \
                --mem=64G \
                --wrap="bash $SCRIPT_DIR/test_models.sh --model '$MODEL_ID'"
        else
            # All models submission
            echo "📝 Submitting test job for all models"
            echo "⏱️  Time limit: $TIME_LIMIT"
            
            sbatch \
                --job-name="test-all-models" \
                --time="$TIME_LIMIT" \
                --gres=gpu:1 \
                --ntasks=1 \
                --cpus-per-task=8 \
                --mem=64G \
                --wrap="bash $SCRIPT_DIR/test_models.sh"
        fi
        
        echo "✅ Job submitted successfully!"
        ;;
        
    interactive)
        # Interactive mode (useful for debugging)
        echo "Running in interactive mode..."
        
        # Load modules if needed
        if [ -f /etc/profile.d/modules.sh ]; then
            . /etc/profile.d/modules.sh
            module load cuda/12.1.0 2>/dev/null || true
        fi
        
        export PATH="$HOME/.local/bin:$PATH"
        
        if [ -n "$MODEL_ID" ]; then
            load_env_file
            test_single_model "$MODEL_ID"
        else
            test_all_models
        fi
        ;;
esac