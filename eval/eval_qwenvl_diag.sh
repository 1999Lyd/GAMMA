#!/bin/bash
# GroundSG+QwenVL tick-diagnosis run — the single-VLM / no-harness baseline
# (RoboMME's released adapter, Qwen3-VL-4B + LoRA checkpoint-1200), driven by
# THEIR eval.py so prompts/history/keep_period semantics are verbatim.
#
# One lane: policy server (same fixed subgoal-conditioned pi0.5, ckpt 79999,
# as GAMMA and the oracle rows) on $GPU_POL; eval.py client on $GPU_CLI runs
# ManiSkill sim + PtEngine 4B in-process (venv: robomme_qwenvl_venv, built
# from examples/robomme/requirements.txt: ms_swift 3.11.1, torch 2.9.1).
#
# WAM_DIAG_TRACE=1 (patch in subgoal_predictor.py) logs per tick
#   {count, pred, oracle}  ->  <save>/diag_<task>_ep<i>.jsonl
# and keeps episode frames (step_*_image.png) for later SAM-3 grounding
# measurement. Results: runs/evaluation/$NAME/ckpt79999/seed$SEED/qwenvl/
#
# Usage: eval_qwenvl_diag.sh [GPU_POL] [GPU_CLI] [EPISODES] [TASKS]
set -u
GPU_POL=${1:-2}; GPU_CLI=${2:-3}
EPISODES=${3:-10}
# diagnosis families: counting/timing (BU,BUS,StopC), binding (VUS,VRP,VPO),
# direction (PL,RS) — the three win narratives of Sec 5.2
TASKS=${4:-ButtonUnmask,ButtonUnmaskSwap,StopCube,VideoUnmaskSwap,VideoRepick,VideoPlaceOrder,PatternLock,RouteStick}
SEED=${SEED:-7}; STEP=79999; NAME=${NAME:-qwenvl_groundsg_diag}
O=${ROBOMME_ROOT}
VENV=/home/user/belief_vla_migration/robomme_qwenvl_venv
L=${GAMMA_LOGS}
HOLD=${GAMMA_ROOT}/eval/gpu_hold.py

find_free_port(){ for i in $(seq 1 500); do p=$(shuf -i 20000-30000 -n1);
  lsof -iTCP:$p -sTCP:LISTEN &>/dev/null || { echo $p; return 0; }; done; return 1; }
wait_port(){ for i in $(seq 1 300); do
  timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null && return 0
  kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }

P=$(find_free_port)
echo "$(date +%F_%T) qwenvl diag  policy gpu$GPU_POL port $P  client gpu$GPU_CLI  eps=$EPISODES seed=$SEED"

cd $O
CUDA_VISIBLE_DEVICES=$GPU_POL nohup setsid .venv/bin/python3 scripts/serve_policy.py \
  --seed=$SEED --port=$P policy:checkpoint \
  --policy.dir=runs/ckpts/mme_vla_suite/symbolic_grounded_repro/$STEP \
  --policy.config=mme_vla_suite > $L/policy_${NAME}.log 2>&1 & SP=$!
wait_port $P $SP || { echo "policy server FAILED (see $L/policy_${NAME}.log)"; exit 1; }
echo "policy server up (pid $SP)"

cd $O/examples/robomme
WAM_DIAG_TRACE=1 CUDA_VISIBLE_DEVICES=$GPU_CLI HF_HOME=${HF_HOME} \
PYTHONPATH=$O/examples/robomme:$O/packages/openpi-client/src \
  $VENV/bin/python -u eval.py \
  --args.host=127.0.0.1 --args.port=$P --args.max-episodes=$EPISODES \
  --args.use-qwenvl --args.subgoal-type=grounded_subgoal \
  --args.only-tasks=$TASKS \
  --args.policy_name=$NAME --args.model_seed=$SEED --args.model_ckpt_id=$STEP \
  > $L/client_${NAME}.log 2>&1
RC=$?
echo "$(date +%F_%T) client done rc=$RC"

kill $SP 2>/dev/null
# reserve the pair for the project (holders self-release to project procs)
for g in $(printf "%s\n%s\n" "$GPU_POL" "$GPU_CLI" | sort -u); do
  CUDA_VISIBLE_DEVICES=$g nohup setsid \
    ${CLIENT_PY} $HOLD 60 \
    > /dev/null 2>&1 &
done
echo "holders armed on gpu$GPU_POL,$GPU_CLI"
