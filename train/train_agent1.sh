set -u
OUT=${GAMMA_DATA}/runs/agent1_v17_lora
D=${GAMMA_DATA}/data/wam_sft_v17
export PATH=${MSSWIFT_BIN}:$PATH
HF_HOME=${HF_HOME} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --model 'Qwen/Qwen3.5-9B' \
    --dataset $D/agent1_swift_train.jsonl \
    --val_dataset $D/agent1_swift_val.jsonl \
    --split_dataset_ratio 0.0 \
    --packing false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_steps 500 \
    --logging_steps 50 \
    --eval_steps 500 \
    --max_length 3200 \
    --output_dir $OUT \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataloader_num_workers 4
