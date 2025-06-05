#!/bin/bash
# export PATH=/usr/local/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# export NCCL_P2P_LEVEL=NVL
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_P2P_LEVEL=2
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_IB_TIMEOUT=22
# export NCCL_BLOCKING_WAIT=0

nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 21055 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path your-vicuna-1.5-7b-path \
    --version plain \
    --data_path pretrain.jsonl \
    --image_folder none \
    --lowres_vision_tower your-clip-large-336-path \
    --highres_vision_tower your-convnext-1536-path \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./pretrained_mm_projector/Pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none > ./logs/Pretrain.log &