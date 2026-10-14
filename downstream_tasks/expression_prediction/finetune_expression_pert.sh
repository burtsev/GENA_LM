#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath "$SCRIPT_DIR/../..")
GENALM_HOME=${GENALM_HOME:-$(realpath "$REPO_ROOT/..")}
PERT_DATA_ROOT=${PERT_DATA_ROOT:-$GENALM_HOME/pert_data/michael}
PERT_GENOME=${PERT_GENOME:-$PERT_DATA_ROOT/genomes/GRCh38.primary_assembly.genome.fa}
VENV_DIR=${VENV_DIR:-$GENALM_HOME/.venv}
ACCELERATE_BIN=${ACCELERATE_BIN:-$VENV_DIR/bin/accelerate}

cd "$REPO_ROOT"

# Use a user CUDA toolkit when it exists. PyTorch wheels can otherwise use
# their bundled CUDA runtime and the system driver.
CUDA_HOME=${CUDA_HOME:-$HOME/.local/cuda}
if [[ -d "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
IFS=',' read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"

TBS=${TBS:-80}
BS=${BS:-1}
NP=${NP:-${#cuda_devices[@]}}
denominator=$((BS * NP))
if (( denominator <= 0 || TBS % denominator != 0 )); then
  echo "TBS ($TBS) must be divisible by BS * NP ($denominator)" >&2
  exit 2
fi
GAS=$((TBS / denominator))

config_name=${CONFIG_NAME:-final_pert}
main_process_port=${MAIN_PROCESS_PORT:-29515}

for required_path in \
  "$PERT_DATA_ROOT/models" \
  "$PERT_DATA_ROOT/data/pert" \
  "$PERT_GENOME"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required perturbation asset is missing: $required_path" >&2
    exit 2
  fi
done
if [[ ! -x "$ACCELERATE_BIN" ]]; then
  echo "Accelerate is missing from $ACCELERATE_BIN; create $VENV_DIR first." >&2
  exit 2
fi

export GENALM_HOME PERT_DATA_ROOT PERT_GENOME
"$ACCELERATE_BIN" launch \
  --main_process_port "$main_process_port" \
  --num_machines 1 \
  --num_processes "$NP" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final_pert \
  --experiment_config "downstream_tasks/expression_prediction/configs/${config_name}.yaml" \
  --batch_size "$BS" \
  --gradient_accumulation_steps "$GAS" \
  "$@"
