#!/usr/bin/env bash
set -euo pipefail

# Executes examples/test_model.py across one or more model IDs using the
# appropriate uv environment group inferred from examples.model_table.
#
# Usage:
#   examples/test_models.sh [--runtime <runtime>] [model_id ...]
#
# Arguments:
#   --runtime <runtime>  Optional runtime override applied to every model
#                        (transformers | vllm | api). Defaults to each model's
#                        configured default runtime.
#   model_id             One or more model identifiers; if omitted, a small
#                        representative set is used.
#
# Environment variables:
#   DEFAULT_TRANSFORMERS_GROUP  Fallback uv group for transformers runtime
#                               (default: normal).
#   DEFAULT_API_GROUP           Fallback uv group for api runtime (default: normal).
#   DEFAULT_VLLM_GROUP          Fallback uv group when vLLM config omits one
#                               (default: vllm_normal).
#   GPU_MEMORY_UTILIZATION      Passed to vLLM base wrapper (default: 0.9).
#   TENSOR_PARALLEL_SIZE        Override auto-detected tensor parallel size
#                               (default: number of CUDA_VISIBLE_DEVICES entries).
#   MODEL_GROUP_OVERRIDES       Comma-separated list of model_id=group pairs to
#                               force a specific uv group, e.g.
#                               "modelA=normal,modelB=phi".
#
# Examples:
#   examples/test_models.sh llava-hf/llava-1.5-7b-hf
#   examples/test_models.sh --runtime vllm Qwen/Qwen3-VL-30B-A3B-Instruct
#   MODEL_GROUP_OVERRIDES="MIL-UT/Asagi-14B=asagi" examples/test_models.sh MIL-UT/Asagi-14B

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/test_models"
SUCCESS_LOG="${LOG_DIR}/success.log"
mkdir -p "${LOG_DIR}"

usage() {
  grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

runtime_override=""
declare -a MODEL_IDS=()

while (($#)); do
  case "$1" in
    --runtime)
      if [[ $# -lt 2 ]]; then
        echo "--runtime requires an argument" >&2
        exit 1
      fi
      runtime_override="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        MODEL_IDS+=("$1")
        shift
      done
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      MODEL_IDS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#MODEL_IDS[@]} -eq 0 ]]; then
  if [[ -n "${runtime_override}" ]]; then
    mapfile -t MODEL_IDS < <(
      RUNTIME_FILTER="${runtime_override}" \
      uv run --group normal python - <<'PY'
import os
from examples.model_table import get_supported_model_ids

runtime = os.environ["RUNTIME_FILTER"]
for model_id in get_supported_model_ids(runtime=runtime):
    print(model_id)
PY
    )
  else
    mapfile -t MODEL_IDS < <(
      uv run --group normal python - <<'PY'
from examples.model_table import get_supported_model_ids

for model_id in get_supported_model_ids():
    print(model_id)
PY
    )
  fi

  if [[ ${#MODEL_IDS[@]} -gt 0 ]]; then
    filtered_ids=()
    for model_id in "${MODEL_IDS[@]}"; do
      [[ -z "${model_id}" ]] && continue
      if [[ "${model_id}" == */* ]]; then
        filtered_ids+=("${model_id}")
      fi
    done
    MODEL_IDS=("${filtered_ids[@]}")
  fi

  if [[ ${#MODEL_IDS[@]} -eq 0 ]]; then
    echo "テスト対象のモデルが見つかりませんでした" >&2
    exit 1
  fi
fi

DEFAULT_TRANSFORMERS_GROUP=${DEFAULT_TRANSFORMERS_GROUP:-normal}
DEFAULT_API_GROUP=${DEFAULT_API_GROUP:-normal}
DEFAULT_VLLM_GROUP=${DEFAULT_VLLM_GROUP:-vllm_normal}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

VISIBLE_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
if [[ -z "${VISIBLE_CUDA_DEVICES}" ]]; then
  VISIBLE_CUDA_DEVICES=1
fi

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  EFFECTIVE_TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
else
  IFS=',' read -ra _cuda_device_list <<< "${VISIBLE_CUDA_DEVICES}"
  EFFECTIVE_TENSOR_PARALLEL_SIZE=${#_cuda_device_list[@]}
  unset IFS
  if (( EFFECTIVE_TENSOR_PARALLEL_SIZE == 0 )); then
    EFFECTIVE_TENSOR_PARALLEL_SIZE=1
  fi
fi

# Parse MODEL_GROUP_OVERRIDES into an associative array.
declare -A MODEL_GROUP_OVERRIDE_MAP=()
IFS=',' read -ra _override_pairs <<< "${MODEL_GROUP_OVERRIDES:-}"
unset IFS
for pair in "${_override_pairs[@]}"; do
  if [[ -z "${pair}" ]]; then
    continue
  fi
  model_id=${pair%%=*}
  group_name=${pair#*=}
  if [[ -z "${model_id}" || -z "${group_name}" || "${model_id}" == "${group_name}" ]]; then
    echo "Ignoring malformed MODEL_GROUP_OVERRIDES entry: ${pair}" >&2
    continue
  fi
  MODEL_GROUP_OVERRIDE_MAP["${model_id}"]="${group_name}"
done

run_python() {
  local runtime_arg=${2:-_}
  uv run --group normal python - "$1" "${runtime_arg}" <<'PY'
import sys
from examples.model_table import get_model_spec

model_id = sys.argv[1]
runtime_hint = sys.argv[2]
if runtime_hint == '_' or runtime_hint == '':
    runtime_hint = None

spec = get_model_spec(model_id)
if runtime_hint is not None and runtime_hint not in spec.runtimes:
    available = ', '.join(sorted(spec.runtimes))
    msg = f"Runtime '{runtime_hint}' is not available for {model_id} (available: {available})"
    raise SystemExit(msg)

runtime = runtime_hint or spec.default_runtime
config = spec.get_runtime_config(runtime)
module_path = config.module_path
env_group = config.env_group or ''

print(f"RUNTIME={runtime}")
print(f"DEFAULT_RUNTIME={spec.default_runtime}")
print(f"ENV_GROUP={env_group}")
print(f"MODULE_PATH={module_path}")
PY
}

for model_id in "${MODEL_IDS[@]}"; do
  echo "==== Testing ${model_id} ===="
  runtime_arg="${runtime_override:-_}"
  if ! metadata=$(run_python "${model_id}" "${runtime_arg}"); then
    echo "Failed to resolve runtime metadata for ${model_id}" >&2
    exit 1
  fi
  eval "${metadata}"

  group_override=""
  if [[ -v MODEL_GROUP_OVERRIDE_MAP["${model_id}"] ]]; then
    group_override="${MODEL_GROUP_OVERRIDE_MAP["${model_id}"]}"
  fi

  case "${group_override:+set}" in
    set)
      env_group="${group_override}"
      ;;
    *)
      if [[ "${RUNTIME}" == "vllm" ]]; then
        env_group="${ENV_GROUP:-${DEFAULT_VLLM_GROUP}}"
      elif [[ "${RUNTIME}" == "api" ]]; then
        env_group="${ENV_GROUP:-${DEFAULT_API_GROUP}}"
      else
        env_group="${ENV_GROUP:-${DEFAULT_TRANSFORMERS_GROUP}}"
      fi
      ;;
  esac

  if [[ -z "${env_group}" ]]; then
    echo "Could not determine uv group for ${model_id} (runtime=${RUNTIME})." >&2
    exit 1
  fi

  cmd=(
    uv run --group "${env_group}" python -m examples.test_model
    --model_id "${model_id}"
  )

  if [[ -n "${runtime_override}" ]]; then
    cmd+=(--runtime "${RUNTIME}")
  fi

  if [[ "${RUNTIME}" == "vllm" && "${MODULE_PATH}" == "examples.runtimes.vllm.base.VLLM" ]]; then
    cmd+=(
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
      --tensor_parallel_size "${EFFECTIVE_TENSOR_PARALLEL_SIZE}"
    )
  fi

  echo "Running (CUDA_VISIBLE_DEVICES=${VISIBLE_CUDA_DEVICES}): ${cmd[*]}"
  if CUDA_VISIBLE_DEVICES=${VISIBLE_CUDA_DEVICES} "${cmd[@]}"; then
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf '%s\tmodel_id=%s\truntime=%s\tgroup=%s\n' "${timestamp}" "${model_id}" "${RUNTIME}" "${env_group}" >>"${SUCCESS_LOG}"
  else
    echo "Model test failed for ${model_id}" >&2
    exit 1
  fi
  echo
done
