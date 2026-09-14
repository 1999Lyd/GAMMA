#!/bin/bash
# PickXtimes rerun after the v25 scope revert (PickX out of the v19 video
# scope): 30 episodes, one seed, fresh lane name laneP so lane W's records
# are untouched and nothing is re-sampled.  Usage: rerun_pickx.sh <GPU> <SEED> [WAIT_PID]
set -u
GPU=$1; SEED=$2; WAITPID=${3:-}
S=${GAMMA_WORK}
P=${GAMMA_ROOT}/serve
if [ -n "$WAITPID" ]; then echo "$(date +%F_%T) seed $SEED: waiting for pid $WAITPID to exit"; while kill -0 $WAITPID 2>/dev/null; do sleep 60; done; sleep 60; fi
grep -q 'return "watch the video" in il$' $P/wam_agent_server.py || { echo "ABORT: scope revert not installed in wam_agent_server.py"; exit 1; }
echo "$(date +%F_%T) seed $SEED gpu $GPU: PickXtimes x30 (laneP)"
TASKS_laneP=PickXtimes TASKS_OVERRIDE=PickXtimes XLA_MEM_FRAC=0.20 WAM_VPO_READER=1 WAM_VRP_READER=1 \
  bash $P/eval_0p8b_v20_lane.sh $GPU on 30 $SEED laneP > $S/eval_0p8b_s${SEED}_laneP.log 2>&1
echo "$(date +%F_%T) seed $SEED laneP rc=$?"
python3 - "$SEED" <<'PY'
import json,sys
p=f"${ROBOMME_ROOT}/examples/robomme/runs/evaluation/wam_0p8b_on20_laneP/ckpt79999/seed{sys.argv[1]}/oracle/log.json"
try: d=json.load(open(p)); print("RESULT seed",sys.argv[1],d.get("success_rate"))
except Exception as e: print("no log.json yet:",e)
PY
