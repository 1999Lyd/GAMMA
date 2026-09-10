#!/bin/bash
# v14 unattended chain (item 10: agent-2 completion registration; stopcube
# anticipation; complete grounded swap targets; NO oversampling).
set -u
S=${GAMMA_WORK}
V=${GAMMA_DATA}/data/wam_sft_v17
RUNS=${GAMMA_DATA}/runs
Q=$S/qvenv/bin/python
export WAM_BASE=Qwen/Qwen3.5-9B
MS=${MSSWIFT_PY}
ST=$S/chain_v17_status.log
say(){ echo "$(date +%F_%T) $1" | tee -a $ST; }

say "STAGE 0: wait for generation"
until grep -q "\[gen\] DONE" $S/gen_v17.log 2>/dev/null; do sleep 60; done
say "STAGE 0 done"

say "STAGE 0.4: per-task training-target videos (16 tasks)"
$Q $S/make_v17_task_videos.py > $S/videos_v14_tasks.log 2>&1 || say "WARN task videos failed"
say "STAGE 0.4 done: $(ls ${GAMMA_DATA}/results/v17_training_videos/*_v14_targets.mp4 2>/dev/null | wc -l) videos"

say "STAGE 1: verification gates"
SPLIT=val $Q $S/audit_derivable.py $V >> $ST 2>&1 || say "WARN audit_derivable failed"
$Q $S/verify_alignment.py $V >> $ST 2>&1 || say "WARN verify_alignment failed"
say "STAGE 1 done"

say "STAGE 2: build agent-1 swift dataset"
SRC=$V $Q $S/make_agent1_swift.py >> $ST 2>&1
say "STAGE 2 done"

say "STAGE 2.5: wait for v13 closed-loop eval to finish, then take GPUs 4-7"
until grep -q "parallel eval done" $S/eval_v16_par.log 2>/dev/null; do sleep 120; done
pkill -f gpu_hold.py || true
sleep 10
say "STAGE 2.5 done (holders released)"

say "STAGE 3: agent-1 training (1 epoch, GPUs 4-7)"
bash $S/train_agent1_v17.sh > $S/train_agent1_v17.log 2>&1
CK=$(ls -d $RUNS/agent1_v17_lora/*/checkpoint-* 2>/dev/null | while read c; do echo "${c##*checkpoint-} $c"; done | sort -n | tail -1 | cut -d' ' -f2)
[ -n "$CK" ] || { say "STAGE 3 FAILED: no checkpoint"; exit 1; }
rm -rf $RUNS/agent1_v17_final; cp -r $CK $RUNS/agent1_v17_final
say "STAGE 3 done: $CK"

say "STAGE 4: agent-1 rollout -> predicted banks (4 shards, GPUs 4-7)"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((4+i)) HF_HOME=${HF_HOME} \
    CKPT=$RUNS/agent1_v17_final SPLIT=train SHARD=$i NSHARD=4 BATCH=24 \
    DATA_DIR=$V FRAMES_DIR=$V \
    nohup $MS -u $S/rollout_agent1.py > $S/rollout_v17_train_s$i.log 2>&1 &
done
wait
cat $V/agent2_pred_train.s{0,1,2,3}.jsonl > $V/agent2_pred_train.jsonl
cat $V/agent1_rollout_train.s{0,1,2,3}.jsonl > $V/agent1_rollout_train.jsonl
CUDA_VISIBLE_DEVICES=4 HF_HOME=${HF_HOME} CKPT=$RUNS/agent1_v17_final \
  SPLIT=val BATCH=24 DATA_DIR=$V FRAMES_DIR=$V \
  $MS -u $S/rollout_agent1.py > $S/rollout_v17_val.log 2>&1
say "STAGE 4 done: $(wc -l < $V/agent2_pred_train.jsonl) train banks"

say "STAGE 5: snap agent-2 targets + build swift data"
$Q $S/snap_agent2_targets.py $V >> $ST 2>&1
SRC=$V BANKSRC=pred TAG=pred $Q $S/make_agent2_swift.py >> $ST 2>&1
say "STAGE 5 done: $(wc -l < $V/agent2_swift_pred_train.jsonl) samples"

say "STAGE 6: agent-2 training (1 epoch, GPUs 4-7)"
bash $S/train_agent2_v17.sh > $S/train_agent2_v17.log 2>&1
CK2=$(ls -d $RUNS/agent2_v17_predbank_lora/*/checkpoint-* 2>/dev/null | while read c; do echo "${c##*checkpoint-} $c"; done | sort -n | tail -1 | cut -d' ' -f2)
[ -n "$CK2" ] || { say "STAGE 6 FAILED: no checkpoint"; exit 1; }
rm -rf $RUNS/agent2_v17_final; cp -r $CK2 $RUNS/agent2_v17_final
say "STAGE 6 done: $CK2"

say "STAGE 7: open-loop agent-2 eval"
CUDA_VISIBLE_DEVICES=4 HF_HOME=${HF_HOME} \
  CKPT=$RUNS/agent2_v17_final VALFILE=$V/agent2_pred_val.jsonl \
  OUTJSON=${GAMMA_DATA}/traces/agent2_openloop_v17.json \
  DATA_DIR=$V \
  $MS $S/eval_agent2.py > $S/eval_agent2_v17.log 2>&1
say "STAGE 7 done: $(grep -A3 'OPEN-LOOP' $S/eval_agent2_v17.log | tail -2 | tr '\n' ' ')"

say "STAGE 8: 2-lane closed-loop eval (10 eps x 16 tasks)"
bash $S/eval_wam_parallel_v13.sh 79999 10 wam_v17_par \
  $RUNS/agent1_v17_final $RUNS/agent2_v17_final > $S/eval_v17_par.log 2>&1
say "STAGE 8 done: $(tail -3 $S/eval_v17_par.log | tr '\n' ' ')"
say "CHAIN V17 COMPLETE"
