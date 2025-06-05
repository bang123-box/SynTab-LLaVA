# SynTab-LLaVA: Enhancing Multimodal Table Understanding with Decoupled Synthesis
[\[Paper\]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_SynTab-LLaVA_Enhancing_Multimodal_Table_Understanding_with_Decoupled_Synthesis_CVPR_2025_paper.pdf)  [\[Weights\]](https://pan.baidu.com/s/1WnJK77zC58ZaWhqBD10kwg?pwd=gwk6) [\[Dataset\]](https://pan.baidu.com/s/1WnJK77zC58ZaWhqBD10kwg?pwd=gwk6)

Our paper addresses the constraints in multimodal table understanding (MTU) data due to limited scale. Instead of the usual approach with multimodal large language models that causes issues like hallucinations and high cost, we design a two-step synthesis framework: table image rendering and Q&A pair generation using table codes and LLMs. This reduces costs and hallucinations while improving accuracy. We synthesize a large-scale dataset SynTab with 636K images and 1.8M samples under $200. Also, we introduce the SynTab-LAVA model which achieves SOTA performance on 21 of 24 benchmarks, showing the effectiveness and generalization of our method.

## [Download our Dataset](https://pan.baidu.com/s/1WnJK77zC58ZaWhqBD10kwg?pwd=gwk6)
You can download the `images_tar`, `sft.json`, and `pretrain.json` files from the file shared via Baidu Netdisk: [SynTab-Data](https://pan.baidu.com/s/1WnJK77zC58ZaWhqBD10kwg?pwd=gwk6). 
Then, extract the 4 tar files in `images_tar` and organize them in the following format. Later, We further expanded the dataset, increasing the SFT data from the original 1.8M to 2.4M.
```
SynTab-Data
├── rel_extraction_train
├── fintab
├── ent_link_train
├── col_type_train
├── sft.json
├── pretrain.json
└── images_tar
    ├── rel_extraction_train.tar.gz
    ├── fintab.tar.gz
    ├── ent_link_train.tar.gz
    └── col_type_train.tar.gz

```

## [Download Checkpoints](https://pan.baidu.com/s/18fdnBRUqjBXYVVGTU8GT2g?pwd=qnva)
```
SynTab-LLaVA-v1.5-7B-Conv-Clip
├── ConvLLaVA-ConvNeXt-1536
├── Clip-vit-largr-patch14-336
├── model-00001-of-00004.safetensors
├── ....
├── model-00004-of-00004.safetensors
├── config.json
├── tokenize_config.json
├── generation_config.json
├── special_tokens_map.json
├── tokenizer.model
└── model.safetensors.index.json
```
Modify the values of `mm_highres_vision_tower` and `mm_lowres_vision_tower` in the `config.json` file to the model path on your local machine.

## Install
1. Install Package
```Shell
git clone https://github.com/bang123-box/SynTab-LLaVA.git
cd SynTab-LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install protobuf
```

2. Install additional packages for training and evalution code
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
cd evalution
pip install -r eval_requirements.txt
cd ..
```

## Train
To ensure the integrity of the experiment, we also need to download some datasets required in [Table-LLaVA](https://github.com/SpursGoZmy/Table-LLaVA). We need to download [enhanced_llava_pretrain_data_708K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/enhanced_llava_pretrain_data_708K.json) required for the pre-training phase in the paper, and [MMTab-instruct_sft_data_llava_format_232K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-instruct_sft_data_llava_format_232K.json) required for the SFT phase. Finally, for testing purposes, we also need to download [MMTab-eval_test_data_49K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab/blob/main/MMTab-eval_test_data_49K.json). For specific data download and organization, please refer to [Table-LLaVA](https://github.com/SpursGoZmy/Table-LLaVA). The final data organization is as follows:
```
data
├── SynTab-Data
|   ├── rel_extraction_train
|   ├── fintab
|   ├── ent_link_train
|   ├── col_type_train
|   ├── sft.json
|   ├── pretrain.json
|   └── images_tar
|       ├── rel_extraction_train.tar.gz
|       ├── fintab.tar.gz
|       ├── ent_link_train.tar.gz
|       └── col_type_train.tar.gz
└── Table-LLaVA
    ├── LLaVA-Pretrain
    |   ├── images
    |   |   ├── table_pretrain_part_1
    |   |   ├── table_pretrain_part_2
    |   |   ├── 00453
    |   |   ├── 00019
    |   |   ├── ...
    |   |   └── 00095
    |   └── enhanced_llava_pretrain_data_708K.json
    ├── LLaVA-Finetune
    |   ├── images
    |   |   ├── table_instructV
    |   └── MMTab-instruct_sft_data_llava_format_232K.json
    └── LLaVA-Inference
        ├── all_test_images
        ├── MMTab-eval_test_tables_23K.json
        ├── MMTab-eval_test_data_49K.json
        └── MMTab-eval_test_data_49K_llava_jsonl_format.jsonl
```

1. Pretraining

First, modify the dialogue data path and image path in the `pretrain.jsonl` file. Then, modify the necessary parameters in the `scripts/v1_5/pretrain.sh` file, including: `model_name_or_path`, `highres_vision_tower`, `lowres_vision_tower`. You can download the models via the following link.

model_name_or_path: [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)

highres_vision_tower: ConvNeXt-1536 (can be found in the [download checkpoints](https://pan.baidu.com/s/18fdnBRUqjBXYVVGTU8GT2g?pwd=qnva))

lowres_vision_tower: clip-vit-large-patch-336 (can be found in the [download checkpoints](https://pan.baidu.com/s/18fdnBRUqjBXYVVGTU8GT2g?pwd=qnva))


```bash
bash scripts/v1_5/pretrain.sh
```


2. SFT

First, modify the dialogue data path and image path in the `sft.jsonl` file. Then, modify the necessary parameters in the `scripts/v1_5/finetune_lora.sh` file, then:

```
bash scripts/v1_5/finetune_lora.sh
```


## Evaluation
The testing steps are similar to those of Table-LLaVA. If there's anything unclear, you can refer to their repo or contact us for guidance.
### Step 1: Modify the Inference Script

modify **scripts/table_llava_scripts/table_llava_inference.sh**：
```
#!/bin/bash

GPULIST='0,1,2,3,4,5,6,7'

IFS=',' read -ra GPULIST <<< "$GPULIST"

CHUNKS=${#GPULIST[@]}

CKPT="Sftlora-SynTab-llava-v1.5-7B-Conv-Clip"
SPLIT="MMTab_eval"
# If it is a LoRA fine-tuned model, `--model-path` should be the path of the LoRA fine-tuned model, and `--model-base` should be the storage path of the large language model (LLM).
# If it is a full fine-tuning, set `--model-path` to the path of the saved model and set `--model-base` to `None`.
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${GPULIST[$IDX]}
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-path ./checkpoints/${CKPT} \
         --model-base your_vicuna1_5-7B-path \
        --question-file your_path/Table-LLaVA/LLaVA-Inference/MMTab-eval_test_data_49K_llava_jsonl_format.jsonl \
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
```

### Step 2: Execute the Inference Script

```bash
bash scripts/table_llava_scripts/start_multicard_inference.sh
```
After this step, you will obtain the prediction results, which will be stored in: `./eval_results/answers/MMTab_eval/Table-Sftlora_llava/merge.jsonl`.

### Step 4: Start Testing
```
cd evaluation
python evaluation.py --input_file ./eval_results/answers/MMTab_eval/Sftlora-SynTab-llava-v1.5-7B-Conv-Clip/merge.jsonl --test_data your_path/Table-LLaVA/LLaVA-Inference/MMTab-eval_test_data_49K.json --test_table your_path/Table-LLaVA/LLaVA-Inference/MMTab-eval_test_tables_23K.json

```

## Acknowledgement
We would like to express our sincere gratitude for the assistance provided by the following open-source repositories:
- [LLaVA](https://github.com/haotian-liu/LLaVA): 
- [Table-LLaVA](https://github.com/SpursGoZmy/Table-LLaVA)
- [Conv-LLaVA](https://github.com/alibaba/conv-llava)

Our code, as well as some of the datasets and models used during training and testing, were made possible thanks to their open-sourcing.
