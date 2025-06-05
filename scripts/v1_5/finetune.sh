#!/bin/bash
# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1


nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 20056 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path your-vicuna-1.5-7b-path \
    --version v1 \
    --data_path sft.jsonl \
    --image_folder none \
    --lowres_vision_tower clip-large-336-path \
    --highres_vision_tower convnext-1536-path \
    --pretrain_mm_mlp_adapter pretrained_mm_projector/Pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Table-Sft_llava \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none > ./logs/Table-Sft_llava.log &
