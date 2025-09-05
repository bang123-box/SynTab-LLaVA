#!/bin/bash
CHUNKS=64

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing chunk $IDX"
    python table_qa_pipeline.py \
        --input_file  /data/Table/pretrain/pretrain_636179.json \
        --output_dir /data/Table/pretrain/gen_qa \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done
