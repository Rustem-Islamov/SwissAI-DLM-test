#!/usr/bin/env bash
set -euo pipefail

# One whole node (4 GPUs) per axis: the (seed, arm, rung) runs are sharded over 4 workers.
# Coordinate checks for the GIDD-HF muP / CompleteP parametrization
# (SwissAI-DLM_coordinate_checks.md).  One single-GPU job per check; each job runs
# both arms (on = completep, off = no depth / width control) for all rungs and seeds.
#
#   AXES="noop"                 test 6b: base rung, on == off (run first; cheap)
#   AXES="width"                per-module act/prop/eff, 4 widths x 4 seeds x 100 steps
#   AXES="depth"                ||h_L^t - h_L^0||, L in {8,16,32,64} at width 512, steps 2..10
#   AXES="cosc"                 w/L = 64, w in {512,768,1024,1536}
#
# Dry run by default; submit with DRY_RUN=0.  Fit afterwards with
#   python scripts/analysis/coordinate_check_fit.py <axis> ${OUT_DIR} [--plot]
# CPU-only checks 6a/6c: tests/test_adamw_mup.py (torch-free classification and scale rules);
# 6c on the real model is also asserted at the start of every run (no parameter may be "other").

DRY_RUN="${DRY_RUN:-1}"
AXES="${AXES:-noop width depth cosc}"
SEEDS="${SEEDS:-1 2 3 4}"
LR="${LR:-2.19116e-2}"            # peak LR at base_width (optim=adamw_mup convention)
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"   # must match a packed dataset (grid script uses 2048)
BASE_WIDTH="${BASE_WIDTH:-256}"
BASE_DEPTH="${BASE_DEPTH:-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"      # forwarded to coordinate_check.py, e.g. "--widths 512 1024 2048 4096"
ACCOUNT="${ACCOUNT:-ab035}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
GPUS_PER_NODE=4                   # whole node per job: one worker per GPU, runs sharded across them

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/venv-docker/bin/python"
DATA_SRC_ROOT="${DATA_SRC_ROOT:-/iopsstor/scratch/cscs/zdelbari/SwissAI-DLM-data}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/iopsstor/scratch/cscs/${USER}}/SwissAI-DLM-data}"
PRETOK_LOCAL_DIR="${PRETOK_LOCAL_DIR:-${DATA_SRC_ROOT}/training-data/gidd-nemotron-cc-pretok}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${DATA_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok}"
OUT_BASE="${OUT_BASE:-${DATA_ROOT}/outputs/coordinate_checks/${EXPERIMENT_NAME:-gidd_hf_ref_completep}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/coordinate_checks}"

# The shared packed dataset for this sequence length is symlinked into the user's cache dir,
# as in submit_adamw_mup_hp_grid.sh, so nothing is written next to the shared data.
link_data() {
  mkdir -p "${DATA_CACHE_DIR}"
  for split in train valid; do
    name="nemotron-cc-pretok-${split}_$([[ ${split} == train ]] && echo train || echo validation)_bs${SEQ_LEN}_unwrapped_eosFalse_specialFalse.dat"
    src="${DATA_SRC_ROOT}/cache/discrete_diffusion/nemotron-cc-pretok/${name}"
    [[ -d "${src}" ]] || { echo "Missing shared packed dataset: ${src} (is SEQ_LEN=${SEQ_LEN} packed?)" >&2; exit 1; }
    [[ -e "${DATA_CACHE_DIR}/${name}" ]] || ln -s "${src}" "${DATA_CACHE_DIR}/${name}"
  done
}

[[ "${DRY_RUN}" == "1" ]] || { link_data; mkdir -p "${LOG_DIR}"; }
PARTITION_DIRECTIVE=""
[[ -n "${PARTITION}" ]] && PARTITION_DIRECTIVE="#SBATCH --partition=${PARTITION}"

for AXIS in ${AXES}; do
  OUT_DIR="${OUT_BASE}/${AXIS}"
  case "${AXIS}" in
    noop)  AXIS_ARGS="--width 768" ;;   # w768_L12: the tuned base model
    width) AXIS_ARGS="--widths 256 512 1024 2048 --depth ${BASE_DEPTH}" ;;
    depth) AXIS_ARGS="--width 512 --depths 8 16 32 64" ;;
    cosc)  AXIS_ARGS="--widths 512 768 1024 1536 --aspect 64" ;;
    *) echo "unknown axis ${AXIS}" >&2; exit 1 ;;
  esac
  CMD="${VENV_PYTHON} -u ${REPO_ROOT}/scripts/analysis/coordinate_check.py --axis ${AXIS} ${AXIS_ARGS} \
    --seeds ${SEEDS} --lr ${LR} --batch-size ${BATCH_SIZE} --seq-len ${SEQ_LEN} \
    --base-width ${BASE_WIDTH} --base-depth ${BASE_DEPTH} \
    --pretok-dir ${PRETOK_LOCAL_DIR} --data-cache-dir ${DATA_CACHE_DIR} \
    --out-dir ${OUT_DIR} ${EXTRA_ARGS}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry run] axis=${AXIS} -> ${OUT_DIR}"
    echo "  ${CMD}" | tr -s ' '
    continue
  fi
  "${SBATCH_BIN}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=coordcheck_${AXIS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --exclusive
#SBATCH --account=${ACCOUNT}
${PARTITION_DIRECTIVE}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_DATASETS_CACHE="\${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/transformers"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${OUT_DIR}"

srun --environment=uni-d2 --ntasks=${GPUS_PER_NODE} --gpus-per-task=1 \\
  bash -c '${CMD} --shard-id \$SLURM_PROCID --num-shards ${GPUS_PER_NODE}'
srun --environment=uni-d2 --ntasks=1 --gpus-per-task=0 "${VENV_PYTHON}" scripts/analysis/coordinate_check_fit.py ${AXIS} "${OUT_DIR}" --plot | tee "${OUT_DIR}/fit.txt"
SBATCH_EOF
done
