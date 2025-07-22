set -x CUDA_HOME /usr/local/cuda
set -x PATH /usr/local/cuda/bin $PATH
set -x LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH
set -x ENV_NAME normal # dev evovlm vilaja sarashina normal old stablevlm phi pixtral calm heron_nvila vllm_normal
for env_name in $ENV_NAME
    uv venv .uv/$env_name-env --python python3.12
    source .uv/$env_name-env/bin/activate.fish
    # Install build dependencies first
    uv pip install setuptools wheel packaging ninja
    # Then install torch and other dependencies
    echo "===> Installing torch and dependencies for $env_name"
    uv pip install torch==2.6.0 psutil
    # Sync with no-build-isolation for flash-attn
    echo "===> Syncing environment $env_name with no-build-isolation"
    uv sync --active
    echo "===> Syncing environment $env_name with build isolation"
    uv sync --group $env_name --active
end

# Refer to this url for handling flash-attn
# https://docs.astral.sh/uv/concepts/projects/config/#build-isolation