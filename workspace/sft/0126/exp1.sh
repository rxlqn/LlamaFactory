#!/bin/bash
# lora
# deepspeed
set -x

MODEL_PATH=/cpfs2/shared/models/Qwen3-VL-2B-Instruct

# llamafactory-cli train \
#     --model_name_or_path ${MODEL_PATH} \
#     --trust_remote_code \
#     --stage sft \
#     --do_train \
#     --finetuning_type lora \
#     --lora_rank 8 \
#     --lora_target all \
#     --dataset identity,alpaca_en_demo \
#     --template qwen3_nothink \
#     --cutoff_len 2048 \
#     --max_samples 1000 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 8 \
#     --output_dir saves/qwen3-4b/lora/sft \
#     --logging_steps 10 \
#     --save_steps 500 \
#     --plot_loss \
#     --overwrite_output_dir \
#     --save_only_model false \
#     --report_to wandb \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 3.0 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.1 \
#     --bf16 \
#     --ddp_timeout 180000000

    # --streaming \
    # --max_steps 10000 \

deepspeed --include localhost:5 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --streaming \
    --dataloader_num_workers 8 \
    --max_steps 10000 \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --dataset ocrsft_3_1_train \
    --template qwen3_vl_nothink \
    --finetuning_type full \
    --output_dir  saves/qwen3-vl-2b/sft \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 