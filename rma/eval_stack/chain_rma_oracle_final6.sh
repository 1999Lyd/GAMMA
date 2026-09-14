#!/bin/bash
# Final protocol eval of rma_pi05_sgprompt_v1 ckpt 79999 (40k updates): 26 tasks
# x 50 trials, oracle subtask AS prompt (ungated feed = their scoring predicates),
# six lanes through the crash-resuming runner, startups 150 s apart, no clients
# or servers on gpu3 (three servers already live there; EGL clients die when
# gpu3 is crowded). Servers: gpu4 x2, gpu6 x2, gpu5, gpu7. Target: GT 46.1/64.8.
set -u
S=${GAMMA_WORK}
O=${ROBOMME_ROOT}; ST=${GAMMA_ROOT}/rma/eval_stack
CFG=rma_pi05_sgprompt; EXP=rma_pi05_sgprompt_v1; STEP=79999; CKPT=$O/runs/ckpts/$CFG/$EXP/$STEP
G4=${GPU4}; G5=${GPU5}
G6=${GPU6}; G7=${GPU7}
echo "$(date +%F_%T) waiting for the $EXP trainer + ckpt $STEP"
while ps -eo cmd | grep -q "[t]rain.py $CFG --exp-name=$EXP"; do sleep 120; done
until [ -f $CKPT/_CHECKPOINT_METADATA ] && [ -d $CKPT/params ]; do sleep 120; done; sleep 60
echo "$(date +%F_%T) launching six 50-trial lanes (ungated oracle feed)"
export RMA_ORACLE=1 RMA_NUM_TRIALS=50 RMA_ACTION_HORIZON=10 RMA_PROMPT_FROM_SUBGOAL=1 RMA_XLA_MEM_FRACTION=0.18 RMA_CKPT_DIR=$CKPT
lane() {  # LANE TS TE PORT SRV CLI
  bash $S/rma_lane_retry.sh $CFG ${EXP}_${STEP} $STEP $2 $3 $4 $5 $6 $ST/outputs/oracle_full50_${EXP}_${STEP}_$1 $S/final50_${EXP}_$1 > $S/lane_final50_$1.log 2>&1
  echo "$(date +%F_%T) final lane $1 finished: $(cat $S/lane_final50_$1.log | tr '\n' ';' | cut -c1-200)"
}
lane L1 1 5   8130 $G4 $G5 & sleep 150; lane L2 6 9   8131 $G4 $G6 & sleep 150; lane L3 10 13 8132 $G6 $G7 & sleep 150
lane L4 14 18 8133 $G6 $G4 & sleep 150; lane L5 19 22 8134 $G5 $G6 & sleep 150; lane L6 23 26 8135 $G7 $G5 & wait
python3 - <<'PY'
import csv, glob
ST='${GAMMA_ROOT}/rma/eval_stack/outputs'; rows={}
for f in glob.glob(f'{ST}/oracle_full50_rma_pi05_sgprompt_v1_79999_L*/task_summary.tsv'):
    for r in csv.DictReader(open(f),delimiter='\t'): rows[int(r['task_id'])]=(float(r['TSR']),float(r['CSR']))
n=len(rows); print(f'FINAL 79999 ungated oracle: {n} tasks x 50 -> TSR {sum(v[0] for v in rows.values())/max(1,n):.1f} / CSR {sum(v[1] for v in rows.values())/max(1,n):.1f}; missing {[t for t in range(1,27) if t not in rows]}')
PY
echo "$(date +%F_%T) final six-lane run done"
