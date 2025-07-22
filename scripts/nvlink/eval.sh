#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# argument
REPO_PATH=$1
model_name=$2
task_name=$3
num_gpus=${4:-1}  # Default to 1 GPU if not specified

# PATH config (additional to config.sh)
export PATH="$HOME/.local/bin:$PATH"

# Load .env file
load_env_file

# Result directories
RESULT_DIR="result"

cd $REPO_PATH
METRIC=${METRIC_MAP[$task_name]}
model_group=${MODEL_GROUP_MAP[$model_name]}
source .uv/$model_group-env/bin/activate

# Run the evaluation and capture exit code
python examples/sample.py \
    --model_id "$model_name" \
    --task_id "$task_name" \
    --metrics "$METRIC" \
    --judge_model "gpt-4.1-2025-04-14" \
    --result_dir "$RESULT_DIR" \
    --overwrite

EXIT_CODE=$?

# Ring the bell with exit code
if [ -f scripts/ring.sh ]; then
    bash scripts/ring.sh $EXIT_CODE
fi

echo "All evaluations are done. Exit code: $EXIT_CODE"

# Exit with the same code as sample.py
exit $EXIT_CODE