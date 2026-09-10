#!/bin/bash
# MemER keyframe-selection audit (intro failure (i)): run RoboMME's released
# MemER baseline (Qwen3-VL-4B + memer/grounded_subgoal checkpoint-1300, their
# eval.py verbatim) with WAM_DIAG_TRACE=1, which logs per tick
#   {count, pred(JSON), oracle, keyframes:[step ids], window:[step ids]}
# and keeps every saved frame so decisive-event windows can be detected
# offline with the v17 SAM-3 pipeline and scored for selection recall /
# retention. Tasks: the decisive-frame families (binding, reveal, press).
# Usage: eval_memer_diag.sh [GPU_POL] [GPU_CLI] [EPISODES]
set -u
GPU_POL=${1:-3}; GPU_CLI=${2:-3}
EPISODES=${3:-10}
TASKS=${4:-VideoUnmaskSwap,VideoRepick,VideoPlaceOrder,PickHighlight,ButtonUnmask,ButtonUnmaskSwap}
SEED=${SEED:-7}; STEP=79999; NAME=${NAME:-memer_diag}
O=${ROBOMME_ROOT}
VENV=${QWENVL_VENV:?set QWENVL_VENV to the benchmark QwenVL venv}
L=${GAMMA_LOGS}

find_free_port(){ for i in $(seq 1 500); do p=$(shuf -i 20000-30000 -n1);
  lsof -iTCP:$p -sTCP:LISTEN &>/dev/null || { echo $p; return 0; }; done; return 1; }
wait_port(){ for i in $(seq 1 300); do
  timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null && return 0
  kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }

P=$(find_free_port)
echo "$(date +%F_%T) memer diag  policy gpu$GPU_POL port $P  client gpu$GPU_CLI  eps=$EPISODES seed=$SEED"

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
  --args.use-memer --args.subgoal-type=grounded_subgoal \
  --args.only-tasks=$TASKS \
  --args.policy_name=$NAME --args.model_seed=$SEED --args.model_ckpt_id=$STEP \
  > $L/client_${NAME}.log 2>&1
RC=$?
echo "$(date +%F_%T) client done rc=$RC"
kill $SP 2>/dev/null
