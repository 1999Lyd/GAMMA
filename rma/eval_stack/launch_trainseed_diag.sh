#!/bin/bash
# DIAGNOSTIC: tasks 1-4 x 10 trials on the authors' default seeds 100-109 (= training layouts)
# for (a) their published pi0.5 with their labels and (b) our ckpt 60000; ungated oracle feed.
set -u
S=${GAMMA_WORK}; ST=${GAMMA_ROOT}/rma/eval_stack
export RMA_ORACLE=1 RMA_NUM_TRIALS=10 RMA_ACTION_HORIZON=10 RMA_PROMPT_FROM_SUBGOAL=1 RMA_XLA_MEM_FRACTION=0.18 RMA_DIAG_TRAIN_SEEDS=100
RMA_PLAN_LABELS=$ST/rma_their_labels.json RMA_CKPT_DIR=${GAMMA_DATA}/ckpts/predimem_hf/vla_alltask \
  bash $S/rma_lane_retry.sh rma_pi05_theirs predimem_vla_alltask theirs 1 4 8180 ${GPU5} ${GPU6} \
  $ST/outputs/TRAINSEEDS_diag10_predimem_vla_alltask_t1-4 $S/probe_TRAINSEEDS_theirs_t1-4 > $S/lane_TRAINSEEDS_theirs.log 2>&1 &
sleep 150
RMA_CKPT_DIR=${ROBOMME_ROOT}/runs/ckpts/rma_pi05_sgprompt/rma_pi05_sgprompt_v1/60000 \
  bash $S/rma_lane_retry.sh rma_pi05_sgprompt rma_pi05_sgprompt_v1 60000 1 4 8181 ${GPU3} ${GPU5} \
  $ST/outputs/TRAINSEEDS_diag10_rma_pi05_sgprompt_v1_60000_t1-4 $S/probe_TRAINSEEDS_ours60000_t1-4 > $S/lane_TRAINSEEDS_ours.log 2>&1 &
wait; echo "$(date +%F_%T) trainseed diag done"
