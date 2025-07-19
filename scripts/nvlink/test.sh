#!/bin/bash
#SBATCH --job-name=llm-jp-test
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# module load
. /etc/profile.d/modules.sh
module load cuda/12.1.0

# argument
REPO_PATH=$1

# PATH config
export PATH="$HOME/.local/bin:$PATH"
export ROOT_DIR="/home/silviase/"
export HF_HOME="$ROOT_DIR/.hf_cache"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/models
export APPTAINER_CACHEDIR="$ROOT_DIR/apptainer_cache"

# NVLink specific configurations
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_NET_GDR_READ=1
export NCCL_P2P_USE_CUDA_MEMCPY=0

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