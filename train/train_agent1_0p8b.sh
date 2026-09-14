#!/bin/bash
# Stage-1 writer (agent-1) on Qwen3.5-0.8B — v17 recipe, batch-64 parity
# (per-device 4 x accum 8 x 2 GPUs). GPUs 2,3 (free; evals own 4-7).
set -u
RUNS=${GAMMA_DATA}/runs
OUT=$RUNS/agent1_0p8b_v17_lora
SW=${MSSWIFT_BIN}/swift
HOLD=${GAMMA_ROOT}/eval/gpu_hold.py
HPY=${CLIENT_PY}

HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
$SW sft \
  --model Qwen/Qwen3.5-0.8B --model_type qwen3_5 --template qwen3_5 \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules all-linear \
  --dataset ${GAMMA_DATA}/data/wam_sft_v17/agent1_swift_train.jsonl \
  --split_dataset_ratio 0 \
  --learning_rate 1e-4 --num_train_epochs 1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --max_length 3200 --torch_dtype bfloat16 --deepspeed zero2 \
  --freeze_vit true --attn_impl sdpa --warmup_ratio 0.05 \
  --dataloader_num_workers 4 --seed 42 \
  --save_steps 500 --output_dir $OUT
RC=$?
echo "$(date +%F_%T) agent1 0.8B sft done rc=$RC"
# reserve the cards after training (kill gpu_hold.py to release)
CUDA_VISIBLE_DEVICES=2 nohup setsid $HPY $HOLD 60 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup setsid $HPY $HOLD 60 > /dev/null 2>&1 &
echo "holders armed on gpu2,3"
