set -x CUDA_HOME /usr/local/cuda
set -x PATH /usr/local/cuda/bin $PATH
set -x LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH
set -x ENV_NAME normal vllm_normal # dev evovlm vilaja sarashina normal old stablevlm phi pixtral calm heron_nvila vllm_normal
for env_name in $ENV_NAME
    uv venv .uv/$env_name-env --python python3.12
    source .uv/$env_name-env/bin/activate.fish
    echo "===> Installingdependencies for $env_name"
    uv sync --group $env_name --active
end

# Refer to this url for handling flash-attn
# https://docs.astral.sh/uv/concepts/projects/config/#build-isolation
