#!/bin/bash
# One eval lane with automatic resume after a client core dump (EGL): reruns
# from the first task missing in <OUTPREFIX>*/task_summary.tsv, up to 2 retries,
# each retry in its own out dir (<OUTPREFIX>_r1, _r2) so nothing is overwritten.
# usage: rma_lane_retry.sh CFG EXP STEP TS TE PORT SRV CLI OUTPREFIX LOGPREFIX
# The RMA_* protocol env vars (RMA_ORACLE, RMA_NUM_TRIALS, RMA_CKPT_DIR, ...) are
# inherited from the caller; RMA_CKPT_DIR must be set explicitly.
set -u
CFG=$1; EXP=$2; STEP=$3; TS=$4; TE=$5; PORT=$6; SRV=$7; CLI=$8; OUTP=$9; LOGP=${10}
ST=${GAMMA_ROOT}/rma/eval_stack
: "${RMA_CKPT_DIR:?set RMA_CKPT_DIR}"
suffix=""; ts=$TS
for attempt in 0 1 2; do
  out=${OUTP}${suffix}
  MME_RMA_PORT=$PORT RMA_TASK_START=$ts RMA_TASK_END=$TE RMA_OUT_ROOT=$out \
    bash $ST/run_rma_eval.sh $CFG ${EXP}_$(basename $out) $STEP $SRV $CLI > ${LOGP}${suffix}.log 2>&1
  echo "$(date +%F_%T) lane $(basename $out) tasks $ts-$TE rc=$? $(tr -d '\n ' < $out/aggregate.json 2>/dev/null | cut -c1-100)"
  missing=$(python3 - "$OUTP" "$ts" "$TE" <<'PY'
import sys, csv, glob
pre, ts, te = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); done = set()
for f in glob.glob(pre + '*/task_summary.tsv'):
    for r in csv.DictReader(open(f), delimiter='\t'): done.add(int(r['task_id']))
m = [t for t in range(ts, te + 1) if t not in done]; print(m[0] if m else '')
PY
)
  [ -z "$missing" ] && break
  ts=$missing; suffix="_r$((attempt+1))"
  echo "$(date +%F_%T) lane $(basename $OUTP): client lost tasks from $ts; retrying as $(basename $OUTP)$suffix"; sleep 20
done
