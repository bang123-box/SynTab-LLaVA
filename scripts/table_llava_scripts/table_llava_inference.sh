#!/bin/bash

GPULIST='0,1,2,3,4,5,6,7'

IFS=',' read -ra GPULIST <<< "$GPULIST"

CHUNKS=${#GPULIST[@]}

CKPT="SynTab-LLaVA-v1.5-7B_Conv-Clip"
SPLIT="MMTab_eval"
# If it is a LoRA fine-tuned model, `--model-path` should be the path of the LoRA fine-tuned model, and `--model-base` should be the storage path of the large language model (LLM).
# If it is a full fine-tuning, set `--model-path` to the path of the saved model and set `--model-base` to `None`.
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${GPULIST[$IDX]}
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-path your_abs_path/SynTab-LLaVA/checkpoints/${CKPT} \
         --model-base your_vicuna1_5-7B-path \
        --question-file your_path/Table-LLaVA/LLaVA-Inference/MMTab-eval_test_data_49K_llava_jsonl_format_shuffle.jsonl \
        --image-folder your_path/Table-LLaVA/LLaVA-Inference/all_test_image \
        --answers-file ./eval_results/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode v1 &
done

wait

output_file=./eval_results/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval_results/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
