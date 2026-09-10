#!/bin/bash
# Parallel WAM closed-loop eval: 2 lanes x 8 tasks on the 4-7 NVLink island.
#   lane A: policy GPU 4, wam-agent GPU 5 (port 8899), client GPU 5
#   lane B: policy GPU 6, wam-agent GPU 7 (port 8898), client GPU 7
# Each lane writes to its own results dir (${NAME}_laneA/_laneB); merge with
# merge_progress.py afterwards.  Task split balances the 9 video-demo tasks
# (long demo phases) across lanes.
# Usage: eval_wam_parallel_v13.sh [STEP] [EPISODES] [NAME] [A1_CKPT] [A2_CKPT]
set -u
STEP=${1:-79999}; EPISODES=${2:-1}; NAME=${3:-wam_twoagent_par}
A1=${4:-${GAMMA_DATA}/runs/agent1_v13_final}
A2=${5:-${GAMMA_DATA}/runs/agent2_v13_final}
S=${GAMMA_WORK}
O=${ROBOMME_ROOT}
UV=$HOME/.local/bin/uv
CLIENT_PY=${CLIENT_PY}
MSPY=${MSSWIFT_PY}
CFG=mme_vla_suite; EXP=symbolic_grounded_repro
L=${GAMMA_LOGS}

# 8+8 split, video-demo tasks (long) balanced 4/5
TASKS_A="BinFill,PickXtimes,ButtonUnmask,ButtonUnmaskSwap,VideoUnmask,VideoRepick,MoveCube,PatternLock"
TASKS_B="StopCube,SwingXtimes,PickHighlight,VideoUnmaskSwap,VideoPlaceButton,VideoPlaceOrder,InsertPeg,RouteStick"

find_free_port(){ for i in $(seq 1 500); do p=$(shuf -i 20000-30000 -n1);
  lsof -iTCP:$p -sTCP:LISTEN &>/dev/null || { echo $p; return 0; }; done; return 1; }

start_agent(){ # $1=gpu $2=port $3=lane
  CUDA_VISIBLE_DEVICES=$1 HF_HOME=${HF_HOME} PORT=$2 \
  A1_CKPT=$A1 A2_CKPT=$A2 \
  WAM_BASE=Qwen/Qwen3.5-9B WAM_TRACE=${GAMMA_DATA}/traces/wam_live_bank_$3.jsonl \
    nohup setsid $MSPY -u $S/wam_agent_server.py > $L/wam_agent_$3.log 2>&1 9>&- &
  echo $!
}

start_policy(){ # $1=gpu $2=port $3=lane
  cd $O
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_DATA_HOME=$O/.openpi_data \
    nohup setsid $UV run scripts/serve_policy.py --seed=7 --port=$2 \
      policy:checkpoint --policy.dir=runs/ckpts/$CFG/$EXP/$STEP --policy.config=$CFG \
      > $L/server_${NAME}_$3.log 2>&1 9>&- &
  echo $!
}

wait_port(){ # $1=port $2=pid
  for i in $(seq 1 300); do
    timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null && return 0
    kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1
}

run_client(){ # $1=gpu $2=policy_port $3=wam_port $4=tasks $5=lane
  cd $O/examples/robomme
  CUDA_VISIBLE_DEVICES=$1 HF_HOME=${HF_HOME} WAM_PORT=$3 \
  PYTHONPATH=$O/examples/robomme:$O/packages/openpi-client/src \
    $CLIENT_PY -u $S/eval_wam_closedloop.py \
      --args.host=127.0.0.1 --args.port=$2 --args.max-episodes=$EPISODES \
      --args.use-oracle --args.subgoal-type=grounded_subgoal \
      --args.only-tasks=$4 \
      --args.policy_name=${NAME}_$5 --args.model_seed=7 --args.model_ckpt_id=$STEP \
      > $L/client_${NAME}_$5.log 2>&1 9>&-
}

PA=$(find_free_port); PB=$(find_free_port)
[ "$PA" = "$PB" ] && PB=$(find_free_port)
echo "$(date +%F_%T) parallel eval step=$STEP eps=$EPISODES  A: gpu6 port $PA  B: gpu7 port $PB (clients gpu5)"

AG_A=$(start_agent 6 8899 laneA); AG_B=$(start_agent 7 8898 laneB)
SP_A=$(start_policy 6 $PA laneA); SP_B=$(start_policy 7 $PB laneB)
wait_port $PA $SP_A || { echo "policy A FAILED"; kill $SP_A $SP_B $AG_A $AG_B 2>/dev/null; exit 1; }
wait_port $PB $SP_B || { echo "policy B FAILED"; kill $SP_A $SP_B $AG_A $AG_B 2>/dev/null; exit 1; }
wait_port 8899 $AG_A || { echo "agent A FAILED"; kill $SP_A $SP_B $AG_A $AG_B 2>/dev/null; exit 1; }
wait_port 8898 $AG_B || { echo "agent B FAILED"; kill $SP_A $SP_B $AG_A $AG_B 2>/dev/null; exit 1; }
echo "all four servers up"

run_client 5 $PA 8899 $TASKS_A laneA & CA=$!
run_client 5 $PB 8898 $TASKS_B laneB & CB=$!
wait $CA; RA=$?; wait $CB; RB=$?
kill $SP_A $SP_B $AG_A $AG_B 2>/dev/null; sleep 8
echo "$(date +%F_%T) parallel eval done  laneA rc=$RA  laneB rc=$RB"

# merged summary
python3 - <<PYEOF
import json, glob
tot=n=0
for lane in ("laneA","laneB"):
    for f in glob.glob(f"$O/examples/robomme/runs/evaluation/${NAME}_"+lane+"/ckpt$STEP/seed7/oracle/progress.json"):
        p=json.load(open(f))
        for t in sorted(p):
            s=sum(p[t].values()); tot+=s; n+=len(p[t])
            print(f"  {t}: {s}/{len(p[t])}")
print(f"TOTAL {tot}/{n} = {tot/max(n,1):.3f}")
PYEOF
