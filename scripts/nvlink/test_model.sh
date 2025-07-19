#!/bin/bash

# Get the repository path automatically
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"

# argument
RUN_MODE=${1:-"sbatch"}  # Default to sbatch mode

# Check run mode
if [ "$RUN_MODE" = "interactive" ]; then
    # Interactive mode - run directly
    echo "Running in interactive mode..."
    
    # module load
    . /etc/profile.d/modules.sh
    module load cuda/12.1.0
    
    # Execute the base script
    bash $(dirname $0)/test_model_base.sh "$REPO_PATH"
    
elif [ "$RUN_MODE" = "sbatch" ]; then
    # Submit mode - submit job
    echo "Submitting job via sbatch..."
    
    # Submit job
    sbatch \
        --job-name=llm-jp-test-models \
        --time=6:00:00 \
        --gres=gpu:1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --wrap="bash $(dirname $0)/test_model_base.sh \"$REPO_PATH\""
    
else
    echo "Invalid run mode: $RUN_MODE"
    echo "Usage: $0 [interactive|sbatch]"
    echo "  interactive: Run interactively"
    echo "  sbatch: Submit to queue (default)"
    exit 1
fi