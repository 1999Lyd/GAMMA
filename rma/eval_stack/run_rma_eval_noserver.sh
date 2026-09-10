#!/usr/bin/env bash
# Closed-loop RoboMemArena 1-26 evaluation for one of our RMA checkpoints.
#
#   ./run_rma_eval.sh <cfg> <exp> <ckpt_id> <server_gpu_uuid> <client_gpu_uuid>
#
# e.g.
#   ./run_rma_eval.sh rma_fs_modul rma_fs_modul_s7 20000 \
#       GPU-2c6cc99e-25c7-05a7-ed4e-83f78a6a75a8 \
#       GPU-cc03980a-3467-e28c-a80e-222d05d31389
#
# Starts scripts/serve_policy.py (training venv, server GPU) on $MME_RMA_PORT,
# waits for the port, then runs their run_all_tasks1_26.py through
# mme_rma_adapter.py in the dedicated eval venv on the client GPU (MuJoCo EGL
# rendering only -- no JAX on the client).
#
# PROTOCOL IS FIXED. Do not edit the flags below without re-reading README.md:
#   --seed 50 --num-trials-per-task 50   (eval layouts 50..99 ONLY)
#   --replan-steps 10 --max-steps 2500 --num-steps-wait 10 --resize-size 256
#   extra-pour rejection ON (harness default)
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "usage: $0 <cfg> <exp> <ckpt_id> <server_gpu_uuid> <client_gpu_uuid>" >&2
    exit 2
fi

CFG="$1"; EXP="$2"; CKPT="$3"; SERVER_GPU="$4"; CLIENT_GPU="$5"

STACK_DIR="${GAMMA_ROOT}/rma/eval_stack"
TRAIN_REPO="${ROBOMME_ROOT}"
BENCH="${RMA_BENCH_ROOT}"
LOG_DIR="${GAMMA_LOGS}"

CKPT_DIR="${RMA_CKPT_DIR:-${TRAIN_REPO}/runs/ckpts/${CFG}/${EXP}/${CKPT}}"
OUT_ROOT="${RMA_OUT_ROOT:-${STACK_DIR}/outputs/${EXP}_${CKPT}}"
LOG_FILE="${LOG_DIR}/rma_eval_${EXP}_${CKPT}.log"
SERVER_LOG="${LOG_DIR}/rma_serve_${EXP}_${CKPT}.log"

# ---- fixed protocol -------------------------------------------------------
SEED=50
NUM_TRIALS="${RMA_NUM_TRIALS:-50}"
if [[ "${NUM_TRIALS}" != 50 ]]; then
    echo "WARNING: num-trials=${NUM_TRIALS} != 50 -> INTEGRATION PROBE, not a benchmark number." >&2
fi
REPLAN_STEPS=10
ACTION_HORIZON="${RMA_ACTION_HORIZON:-20}"
MAX_STEPS=2500
NUM_STEPS_WAIT=10
RESIZE_SIZE=256
TASK_START="${RMA_TASK_START:-1}"
TASK_END="${RMA_TASK_END:-26}"
# "harness" = their build_eval26_policy_input verbatim (the only protocol-legal
# value). "robosuite" is an OFF-PROTOCOL diagnostic for the axis-angle
# convention mismatch documented in README.md / mme_rma_adapter.py.
STATE_CONVENTION="${RMA_STATE_CONVENTION:-harness}"
if [[ "${STATE_CONVENTION}" != "harness" ]]; then
    echo "WARNING: state_convention=${STATE_CONVENTION} -> OFF-PROTOCOL diagnostic run, not a benchmark number." >&2
fi

if (( SEED >= 100 )); then
    echo "FATAL: seed ${SEED} >= 100 -- those are TRAINING layouts. Refusing." >&2
    exit 1
fi
if (( SEED + NUM_TRIALS > 100 )); then
    echo "FATAL: seed range ${SEED}..$((SEED + NUM_TRIALS - 1)) leaks into training layouts (>=100). Refusing." >&2
    exit 1
fi

# ---- connection -----------------------------------------------------------
export MME_RMA_HOST="${MME_RMA_HOST:-127.0.0.1}"
export MME_RMA_PORT="${MME_RMA_PORT:-8105}"
SERVE_SEED="${RMA_SERVE_SEED:-7}"
XLA_FRACTION="${RMA_XLA_MEM_FRACTION:-0.35}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# (noserver variant: checkpoint dir not required -- an already-running policy server on MME_RMA_PORT is reused)

# ---- EGL device for the client -------------------------------------------
# MuJoCo picks the EGL device with MUJOCO_EGL_DEVICE_ID (an *index*, not a
# UUID), and robosuite asserts that the index string occurs inside
# CUDA_VISIBLE_DEVICES. Resolve the UUID to its nvidia-smi index and keep the
# UUID form only when that assert would hold.
CLIENT_IDX="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v u="${CLIENT_GPU}" '$2==u {print $1}')"
if [[ -z "${CLIENT_IDX}" ]]; then
    echo "FATAL: could not resolve client GPU uuid ${CLIENT_GPU} to an index." >&2
    exit 1
fi
if [[ "${CLIENT_GPU}" == *"${CLIENT_IDX}"* ]]; then
    CLIENT_CVD="${CLIENT_GPU}"
else
    # robosuite's substring assert would fail on the UUID form; the client only
    # renders through EGL, so the numeric index is an equivalent pin.
    CLIENT_CVD="${CLIENT_IDX}"
fi

echo "=== RMA eval ==================================================" | tee "${LOG_FILE}"
echo "cfg=${CFG} exp=${EXP} ckpt=${CKPT}"                              | tee -a "${LOG_FILE}"
echo "ckpt_dir=${CKPT_DIR}"                                            | tee -a "${LOG_FILE}"
echo "out_root=${OUT_ROOT}"                                            | tee -a "${LOG_FILE}"
echo "server_gpu=${SERVER_GPU}"                                        | tee -a "${LOG_FILE}"
echo "client_gpu=${CLIENT_GPU} (egl idx ${CLIENT_IDX})"                | tee -a "${LOG_FILE}"
echo "port=${MME_RMA_PORT} seed=${SEED} trials=${NUM_TRIALS} tasks=${TASK_START}..${TASK_END}" | tee -a "${LOG_FILE}"
echo "===============================================================" | tee -a "${LOG_FILE}"

# ---- 1. policy server: REUSED (noserver variant) --------------------------
# This variant attaches to a policy server already listening on
# ${MME_RMA_PORT} (e.g. the 20k server whose checkpoint dir was garbage-
# collected by the resumed trainer's checkpoint manager). It never starts or
# stops a server.
SERVER_PID=""
if ! curl -sf "http://${MME_RMA_HOST}:${MME_RMA_PORT}/healthz" >/dev/null 2>&1; then
    echo "FATAL: no policy server answering on ${MME_RMA_HOST}:${MME_RMA_PORT}" | tee -a "${LOG_FILE}"
    exit 1
fi
echo "[run_rma_eval_noserver] reusing policy server on port ${MME_RMA_PORT}" | tee -a "${LOG_FILE}"

# ---- 2. benchmark sweep ---------------------------------------------------
# RMA_ORACLE=1: privileged grounded-subgoal oracle feeds the S1 policy
# (rma_eval_stack/run_rma_oracle_eval.py; same loop/scoring/outputs).
if [[ "${RMA_ORACLE:-0}" == "1" ]]; then
    CLIENT_SCRIPT="${STACK_DIR}/run_rma_oracle_eval.py"
    echo "[run_rma_eval] ORACLE-FED run (grounded subgoals from sim state)" | tee -a "${LOG_FILE}"
else
    CLIENT_SCRIPT="scripts/run_all_tasks1_26.py"
fi
cd "${BENCH}"
CUDA_VISIBLE_DEVICES="${CLIENT_CVD}" \
MUJOCO_EGL_DEVICE_ID="${CLIENT_IDX}" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
LIBERO_CONFIG_PATH="${STACK_DIR}/libero_config" \
MME_RMA_HOST="${MME_RMA_HOST}" \
MME_RMA_PORT="${MME_RMA_PORT}" \
"${STACK_DIR}/.venv/bin/python" "${CLIENT_SCRIPT}" \
    --adapter-spec "${STACK_DIR}/mme_rma_adapter.py:build_adapter" \
    --adapter-kwargs "{\"host\": \"${MME_RMA_HOST}\", \"port\": ${MME_RMA_PORT}, \"state_convention\": \"${STATE_CONVENTION}\", \"action_horizon\": ${ACTION_HORIZON}}" \
    --task-start "${TASK_START}" \
    --task-end "${TASK_END}" \
    --num-trials-per-task "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --num-steps-wait "${NUM_STEPS_WAIT}" \
    --resize-size "${RESIZE_SIZE}" \
    --fail-on-extra-pour \
    --out-root "${OUT_ROOT}" 2>&1 | tee -a "${LOG_FILE}"

echo "[run_rma_eval] done. results in ${OUT_ROOT}" | tee -a "${LOG_FILE}"
