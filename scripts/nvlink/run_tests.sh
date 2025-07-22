#!/bin/bash
#SBATCH --job-name=llm-jp-pytest
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

# module load
. /etc/profile.d/modules.sh
module load cuda/12.1.0

# argument
REPO_PATH=${1:-$REPO_PATH}

# PATH config (additional to config.sh)
export PATH="$HOME/.local/bin:$PATH"


set -eux

# First change to the repository directory
cd $REPO_PATH

# Then activate the dev environment for testing
source .uv/dev-env/bin/activate
uv sync --active

# Run tests for tasks
echo "Testing tasks..."
uv run --active pytest src/eval_mm/tasks/*.py -v

# Run tests for metrics
echo "Testing metrics..."
uv run --active pytest src/eval_mm/metrics/*.py -v

echo "All tests completed."