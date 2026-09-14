#!/bin/bash
# THEIR published pi0.5, our protocol (seeds 50-99, 50 trials), ungated oracle feed with their labels:
# six memory-gated lanes (server GPU must be below 95 GB before a lane starts), 150 s apart.
set -u
S=${GAMMA_WORK}; ST=${GAMMA_ROOT}/rma/eval_stack
export RMA_ORACLE=1 RMA_NUM_TRIALS=50 RMA_ACTION_HORIZON=10 RMA_PROMPT_FROM_SUBGOAL=1 RMA_XLA_MEM_FRACTION=0.18
export RMA_PLAN_LABELS=$ST/rma_their_labels.json RMA_CKPT_DIR=${GAMMA_DATA}/ckpts/predimem_hf/vla_alltask
waitgpu() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 95000 ]; do sleep 60; done; }
lane() { # TS TE PORT SRVIDX SRV CLI
  waitgpu $4
  bash $S/rma_lane_retry.sh rma_pi05_theirs predimem_vla_alltask theirs $1 $2 $3 $5 $6 $ST/outputs/theirs50u_predimem_vla_alltask_t$1-$2 $S/probe_theirs50u_t$1-$2 > $S/lane_theirs50u_t$1-$2.log 2>&1
}
lane 1 5   8221 4 ${GPU4} ${GPU5} & sleep 150
lane 6 9   8222 4 ${GPU4} ${GPU6} & sleep 150
lane 10 13 8223 3 ${GPU3} ${GPU5} & sleep 150
lane 14 18 8224 5 ${GPU5} ${GPU7} & sleep 150
lane 19 22 8225 6 ${GPU6} ${GPU4} & sleep 150
lane 23 26 8226 7 ${GPU7} ${GPU4} & wait
echo "$(date +%F_%T) theirs50u lanes done"
