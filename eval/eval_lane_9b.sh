#!/bin/bash
# 9B GAMMA closed-loop eval (v17 final ckpts; WAM_A2_IMG=1 for the frame-trained a2), ONE lane on ONE GPU (policy + agent server +
# sim client). Full 16 tasks sequential. ARM=on|off selects the harness
# state (off => WAM_HARNESS_OFF=1 on the agent server).
# Usage: eval_0p8b_lane.sh GPU ARM [EPISODES] [SEED]
set -u
GPU=$1; ARM=$2; EPISODES=${3:-30}; SEED=${4:-7}; LANE=${5:-laneA}
STEP=79999; NAME=wam_9b_${ARM}_${LANE}
TASKS_laneA=BinFill,PickXtimes,ButtonUnmask,ButtonUnmaskSwap,VideoUnmask,VideoRepick,MoveCube,PatternLock
TASKS_laneB=StopCube,SwingXtimes,PickHighlight,VideoUnmaskSwap,VideoPlaceButton,VideoPlaceOrder,InsertPeg,RouteStick
TASKS_laneW=BinFill,PickXtimes,ButtonUnmask,ButtonUnmaskSwap
TASKS_laneX=VideoUnmask,VideoRepick,MoveCube,PatternLock
TASKS_laneY=StopCube,SwingXtimes,PickHighlight,VideoUnmaskSwap
TASKS_laneZ=VideoPlaceButton,VideoPlaceOrder,InsertPeg,RouteStick
O=${ROBOMME_ROOT}
S7=${GAMMA_WORK}
P17=${GAMMA_ROOT}/serve
CLIENT_PY=${CLIENT_PY}
MSPY=${MSSWIFT_PY}
L=${GAMMA_LOGS}
eval "TASKS=\$TASKS_${LANE}"
[ -n "${TASKS_OVERRIDE:-}" ] && TASKS=$TASKS_OVERRIDE

find_free_port(){ for i in $(seq 1 500); do p=$(shuf -i 20000-30000 -n1);
  lsof -iTCP:$p -sTCP:LISTEN &>/dev/null || { echo $p; return 0; }; done; return 1; }
wait_port(){ for i in $(seq 1 360); do
  timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null && return 0
  kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }

PP=$(find_free_port); PA=$(find_free_port); [ "$PP" = "$PA" ] && PA=$(find_free_port)
HOFF=0; [ "$ARM" = "off" ] && HOFF=1
echo "$(date +%F_%T) 0.8B lane arm=$ARM gpu=$GPU policy:$PP agent:$PA eps=$EPISODES seed=$SEED"

CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${HF_HOME} PORT=$PA \
  WAM_BASE=Qwen/Qwen3.5-9B WAM_A2_IMG=1 \
  A1_CKPT=${GAMMA_DATA}/runs/agent1_v17_final \
  A2_CKPT=${GAMMA_DATA}/runs/agent2_v17_final \
  WAM_HARNESS_OFF=$HOFF \
  WAM_TRACE=${GAMMA_DATA}/traces/wam_live_bank_9b_${ARM}_${LANE}.jsonl \
  nohup setsid $MSPY -u $P17/wam_agent_server.py > $L/agent_${NAME}.log 2>&1 & AG=$!

cd $O
CUDA_VISIBLE_DEVICES=$GPU nohup setsid .venv/bin/python3 scripts/serve_policy.py \
  --seed=$SEED --port=$PP policy:checkpoint \
  --policy.dir=runs/ckpts/mme_vla_suite/symbolic_grounded_repro/$STEP \
  --policy.config=mme_vla_suite > $L/policy_${NAME}.log 2>&1 & SP=$!
wait_port $PP $SP || { echo "policy FAILED"; kill $AG 2>/dev/null; exit 1; }
wait_port $PA $AG || { echo "agent FAILED"; kill $SP 2>/dev/null; exit 1; }
echo "servers up (policy $SP, agent $AG)"

cd $O/examples/robomme
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${HF_HOME} WAM_PORT=$PA \
PYTHONPATH=$O/examples/robomme:$O/packages/openpi-client/src \
  $CLIENT_PY -u $S7/eval_wam_closedloop.py \
  --args.host=127.0.0.1 --args.port=$PP --args.max-episodes=$EPISODES \
  --args.use-oracle --args.subgoal-type=grounded_subgoal \
  --args.only-tasks=$TASKS \
  --args.policy_name=$NAME --args.model_seed=$SEED --args.model_ckpt_id=$STEP \
  > $L/client_${NAME}.log 2>&1
RC=$?
echo "$(date +%F_%T) client done rc=$RC"
kill $SP $AG 2>/dev/null
