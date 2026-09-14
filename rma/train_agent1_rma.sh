#!/bin/bash
# RMA writer: Qwen3.5-9B LoRA, the released agent recipe (r16 a32 lr1e-4 1 epoch
# eff. batch 64), one GPU (gpu3). Usage: train_agent1_rma9b.sh <GPU>
set -u
GPU=${1:-3}; RUNS=${GAMMA_DATA}/runs; OUT=$RUNS/agent1_rma_v1_lora
SW=${MSSWIFT_BIN}/swift; F=${GAMMA_DATA}/data/rma_sft_v1
echo "$(date +%F_%T) agent1 RMA 9B sft on gpu$GPU -> $OUT"
HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=$GPU NPROC_PER_NODE=1 \
$SW sft \
  --model Qwen/Qwen3.5-9B --model_type qwen3_5 --template qwen3_5 \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear \
  --dataset $F/agent1_swift_train.jsonl --split_dataset_ratio 0 \
  --learning_rate 1e-4 --num_train_epochs 1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
  --max_length 3200 --torch_dtype bfloat16 --gradient_checkpointing true \
  --freeze_vit true --attn_impl sdpa --warmup_ratio 0.05 \
  --dataloader_num_workers 4 --seed 42 --save_steps 200 --output_dir $OUT
echo "$(date +%F_%T) agent1 RMA sft done rc=$?"
