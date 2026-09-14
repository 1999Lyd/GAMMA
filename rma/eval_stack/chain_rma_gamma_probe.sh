#!/bin/bash
# GAMMA closed-loop probe on RoboMemArena (predicate-only harness unless A1_CKPT
# is given): one lane = its own agent server + policy server + client.
# Usage: chain_rma_gamma_probe.sh <STEP> <L1|L2> [TRIALS=10] [TAG=pred]
set -u
STEP=${1:?ckpt}; LANE=${2:?L1|L2}; TRIALS=${3:-10}; TAG=${4:-pred}
S=${GAMMA_WORK}; PR=${GAMMA_ROOT}/rma
ST=${GAMMA_ROOT}/rma/eval_stack; O=${ROBOMME_ROOT}
CFG=${CFG_OVERRIDE:-rma_pi05_sgprompt}; EXP=${EXP_OVERRIDE:-rma_pi05_sgprompt_v1}; CKPT=${CKPT_OVERRIDE:-$O/runs/ckpts/$CFG/$EXP/$STEP}
G3=${GPU3}; G5=${GPU5}
case $LANE in L1) TS=1; TE=13; APORT=8141; PPORT=8151; SRV=$G3; CLI=$G5 ;; L2) TS=14; TE=26; APORT=8142; PPORT=8152; SRV=$G5; CLI=$G3 ;;
  L2c) TS=19; TE=26; APORT=8143; PPORT=8153; SRV=$G3; CLI=$G5 ;; *) echo bad lane; exit 2 ;; esac
# clients rendering on gpu3 while the SAM-3 cache workers run there core-dump silently (oracle L2 task 18, GAMMA L2 task 19); L2c keeps the client on gpu5
[ -n "${TS_OVERRIDE:-}" ] && TS=$TS_OVERRIDE; [ -n "${TE_OVERRIDE:-}" ] && TE=$TE_OVERRIDE
[ -n "${SRV_OVERRIDE:-}" ] && SRV=$SRV_OVERRIDE; [ -n "${CLI_OVERRIDE:-}" ] && CLI=$CLI_OVERRIDE; AGPU=${AGPU:-3}
[ -n "${APORT_OVERRIDE:-}" ] && APORT=$APORT_OVERRIDE; [ -n "${PPORT_OVERRIDE:-}" ] && PPORT=$PPORT_OVERRIDE
OUT=$ST/outputs/gamma_${TAG}_${EXP}_${STEP}_t${TRIALS}_$LANE; mkdir -p $OUT
AGENT_ENV="WAM_A1_OFF=1 WAM_SAM_OFF=1"; [ -n "${A1_CKPT:-}" ] && AGENT_ENV="A1_CKPT=$A1_CKPT WAM_BASE=${WAM_BASE:-Qwen/Qwen3.5-9B}"
env CUDA_VISIBLE_DEVICES=$AGPU $AGENT_ENV PORT=$APORT WAM_TRACE=$OUT/agent_trace.jsonl \
  setsid nohup ${MSSWIFT_PY} -u $PR/rma_agent_server.py > $OUT/agent_server.log 2>&1 < /dev/null &
AG=$!
for i in $(seq 1 120); do sleep 5; grep -q "rma-agent\] listening on port $APORT pid $AG" $OUT/agent_server.log && break; grep -q -i "traceback\|address already in use" $OUT/agent_server.log && break; done
grep -q "rma-agent\] listening on port $APORT pid $AG" $OUT/agent_server.log || { echo "$(date +%F_%T) agent server failed to bind port $APORT (owner: $(ss -ltnp 2>/dev/null | grep ":$APORT " | grep -o 'pid=[0-9]*'))"; tail -5 $OUT/agent_server.log; kill $AG 2>/dev/null; exit 1; }
echo "$(date +%F_%T) lane $LANE: agent server pid $AG port $APORT; tasks $TS-$TE x $TRIALS on ckpt $STEP"
cd $ST && RMA_GAMMA=1 RMA_AGENT_PORT=$APORT RMA_NUM_TRIALS=$TRIALS RMA_TASK_START=$TS RMA_TASK_END=$TE RMA_ACTION_HORIZON=10 RMA_PROMPT_FROM_SUBGOAL=1 \
  RMA_XLA_MEM_FRACTION=0.18 MME_RMA_PORT=$PPORT RMA_CKPT_DIR=$CKPT RMA_OUT_ROOT=$OUT \
  bash run_rma_eval.sh $CFG gamma_${TAG}_${STEP}_$LANE $STEP $SRV $CLI > $OUT/lane.log 2>&1
echo "$(date +%F_%T) lane $LANE rc=$? $(tr -d '\n ' < $OUT/aggregate.json 2>/dev/null | cut -c1-120)"
kill $AG 2>/dev/null; echo "$(date +%F_%T) agent server $AG stopped"
