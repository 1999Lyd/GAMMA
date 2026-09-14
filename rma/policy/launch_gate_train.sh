#!/bin/bash
# Canonical training launcher for the gate program. Bakes in every env
# requirement that the post-reboot crashes exposed one by one:
#   - WANDB_MODE=offline        (no wandb credentials on this box)
#   - OPENPI_DATA_HOME=.openpi_data  (pi05_base weights live repo-local)
#   - --dataset-path=data/robomme_preprocessed_data
# Usage: bash scripts/launch_gate_train.sh <config> <exp_name> <gpus e.g. 2,3>
set -eu
CFG=${1:?config}; EXP=${2:?exp_name}; GPUS=${3:?gpus}
R=/home/user/belief_vla_migration
# NVLINK ISLAND GUARD : the 8-GPU node's 8 H200s sit on TWO
# NVLink domains, cards 0-3 and cards 4-7. A distributed job spanning both
# falls back to PCIe -- much slower, and a suspected trigger of this node's
# driver instability. Refuse mixed sets rather than train a slow/unstable arm.
# Static UUID->island table on purpose: nvidia-smi blocks in D-state during a
# wedge, so this guard must never call it.
_island () {  # echoes A, B, or ? for one index-or-UUID token
  case "$1" in
    0|1|2|3) echo A ;; 4|5|6|7) echo B ;;
    *4c25b0eb*|*cc03980a*|*063b5691*|*173015bb*) echo A ;;
    *b64ad608*|*2c6cc99e*|*2cf0f2d4*|*2f66c11f*) echo B ;;
    *) echo '?' ;;
  esac
}
_seen=""
for _g in ${GPUS//,/ }; do _i=$(_island "$_g")
  case "$_seen" in *"$_i"*) ;; *) _seen="$_seen$_i" ;; esac
done
case "$_seen" in
  *A*B*|*B*A*) echo "ABORT: GPU set '$GPUS' spans both NVLink islands (0-3 and 4-7)."
               echo "       Pick a pair inside one island (gate arms use cards 4,5)."; exit 2 ;;
  *'?'*)       echo "ABORT: unrecognised GPU token in '$GPUS' — cannot verify NVLink island."; exit 2 ;;
esac
O=$R/robomme_policy_learning_official
MODE=--overwrite
EXTRA=""
# 20k screening sweep (user 2026-07-23): STEPS env caps num_train_steps
[ -n "${STEPS:-}" ] && EXTRA="--num-train-steps=$STEPS"
# SEED env (2026-07-31, seed-replication programme): overrides TrainConfig.seed
# (default 42 -- data order + init rng). Optional; unset keeps prior behavior.
[ -n "${SEED:-}" ] && EXTRA="$EXTRA --seed=$SEED"
WORKERS=${WORKERS:-16}   # loader-bound at 4 (2026-08-22: latent arms 4.3s/it, GPU idle); 256 cores
[ -n "${WORKERS:-}" ] && EXTRA="$EXTRA --num-workers=$WORKERS"
# ACCUM env (user 2026-07-24): micro-batch = 64/ACCUM, effective batch 64;
# micro-step counters scale by ACCUM (ckpt ids too: final = STEPS*ACCUM-1)
# ACCUM_UP (2026-08-01, RMA protocol alignment): micro-batch stays 64,
# effective batch = 64*ACCUM_UP, optimizer steps = STEPS. Micro-step counters
# scale by ACCUM_UP (num-train-steps, save/keep, ckpt ids). LR schedule and
# EMA count OPTIMIZER steps (optax.MultiSteps), so warmup 10k matches the
# RMA authors' 10k optimizer-step warmup exactly.
if [ -n "${ACCUM_UP:-}" ] && [ "${ACCUM_UP}" -gt 1 ]; then
  EXTRA="--num-train-steps=$(( ${STEPS:-40000} * ACCUM_UP )) \
--grad-accum-steps=$ACCUM_UP --save-interval=$((10000 * ACCUM_UP)) --keep-period=$((10000 * ACCUM_UP))"
elif [ -n "${ACCUM:-}" ] && [ "${ACCUM}" -gt 1 ]; then
  MB=$((64 / ACCUM))
  EXTRA="--num-train-steps=$(( ${STEPS:-80000} * ACCUM )) --batch-size=$MB \
--grad-accum-steps=$ACCUM --save-interval=$((10000 * ACCUM)) --keep-period=$((10000 * ACCUM))"
fi
if [ "${RESUME:-0}" = "1" ]; then
  # their trainer needs the explicit step: --resume alone builds .../None/params
  STEP=$(ls $O/runs/ckpts/$CFG/$EXP 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [ -n "$STEP" ]; then
    MODE=--resume; EXTRA="$EXTRA --resum-ckpt-id=$STEP"
  else
    # RESUME=1 on a BRAND-NEW arm: there is nothing to resume from, and
    # --resume with no ckpt id is exactly the .../None/params crash. The queue
    # always passes RESUME=1, so a new arm must fall back to a fresh start.
    echo "no checkpoint for $CFG/$EXP -- starting fresh (--overwrite)"
  fi
fi
cd $O
FSDP=${FSDP:-2}
# DATASET PATH follows the config family (2026-08-01: the hardcoded robomme
# path silently overrode the rma_* configs' default -- an rma arm would have
# trained on RoboMME data; caught at launch).
case "$CFG" in
  rma_*) DSPATH=data/rma_preprocessed_data ;;
  *)     DSPATH=data/robomme_preprocessed_data ;;
esac
# FROZEN-GATE GENERATION. src/mme_vla_suite/shared/frz_gate.py reads these two
# env vars and SILENTLY DEFAULTS TO v3 if they are unset -- the old gpu_queue_*
# scripts set them inline, so a hand launch (or the watchdog) would have swapped
# the gate generation without a word and invalidated the sweep. Set them here,
# once, and abort if the artifacts are missing rather than fall back.
SCR=$R/robomme_setup/scratch
GATE_GEN=${GATE_GEN:-v6}
GATE_HEAD=$SCR/gate_head_${GATE_GEN}_jax.npz
GATE_MU=$SCR/frz_mu_lookup_${GATE_GEN}.npz
case "$CFG" in *frzgate*)
  [ -f "$GATE_HEAD" ] || { echo "ABORT: missing gate head $GATE_HEAD"; exit 3; }
  [ -f "$GATE_MU" ]   || { echo "ABORT: missing mu lookup $GATE_MU"; exit 3; }
  echo "gate generation: $GATE_GEN ($GATE_HEAD)" ;;
esac
MME_FRZ_GATE_HEAD=$GATE_HEAD \
MME_FRZ_MU_LOOKUP=$GATE_MU \
WANDB_MODE=offline \
OPENPI_DATA_HOME=$O/.openpi_data \
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOC:-false} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEMFRAC:-0.9} \
CUDA_VISIBLE_DEVICES=$GPUS \
nohup setsid $HOME/.local/bin/uv run scripts/train.py "$CFG" \
  --exp-name="$EXP" --fsdp-devices=$FSDP $MODE $EXTRA \
  --dataset-path=$DSPATH \
  >> $R/logs/train_${EXP}.log 2>&1 9>&- 8>&- &
echo "$(date +%F_%T) launched $CFG/$EXP pid $! gpus $GPUS"
