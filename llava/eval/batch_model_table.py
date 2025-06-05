import argparse
import torch
import re
import os
import json
from tqdm import tqdm
from typing import Dict, List, Union
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class LlaVaProcessor:
    def __init__(self, tokenizer, image_processor, model_config, mm_use_im_start_end, conv_mode):
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.config = model_config
        self.image_processor = image_processor
        self.conv_mode = conv_mode

    def load_demo_images(image_files: Union[List[str], str]):
        if type(image_files) is list:
            out = []
            for image_file in image_files:
                image = Image.open(image_file).convert("RGB")
                out.append(image)
        else:
            out = [Image.open(image_files).convert("RGB")]
        return out

    # TODO: refactor this, not working
    def get_processed_tokens_demo(self, text: str, image_files: Union[List[str], str]):
        if self.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = self.load_demo_images(image_files)
        image_tensor = torch.stack(
            [self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        )

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        return image_tensor, input_ids

    def format_text(self, text: str):
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens(self, text: str, image_path: str):
        prompt = self.format_text(text)
        image = self.load_image(image_path)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
        prompt = [self.format_text(text) for text in batch_text]
        images = [self.load_image(image_path) for image_path in image_paths]

        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in batch_input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len, self.tokenizer.pad_token_id) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)

        batch_image_tensor = process_images(images, self.image_processor, self.config) #(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids


def get_answerlist(output: str):
    answer_list = []
    matches = re.findall(r'{"answer":\s*"([^"]*)"}', output)
    if matches:
        for match in matches:
            try:
                match = json.loads("{\"answer\":" + match + "}")
                for k, v in match.items():
                    if isinstance(v, list):
                        answer_list.extend(v)
                    elif isinstance(match, str):
                        answer_list.append(v)
                    else:
                        print(f"Unknown type: {type(v)}, only supports list and str.")
            except Exception as e:
                if match[0] == "[" and match[-1] == "]":
                    match = match[1:-1]
                answer_list.append(match)
    else:
        answer_list = [output]
    return answer_list


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    llavaprocessor = LlaVaProcessor(tokenizer, image_processor, model.config, model.config.mm_use_im_start_end, args.conv_mode)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    ans_file = open(answers_file, "w")
    for i in tqdm(range(0, len(questions)), args.bs):
        batch = questions[i : i + args.bs]
        batch_text = [question["question"] for question in batch]
        batch_image_paths = [os.path.join(args.image_folder, question["image"]) for question in batch]
        batch_image_tensor, batch_input_ids = llavaprocessor.get_processed_tokens_batch(batch_text, batch_image_paths)
        stopping_criteria = (
        [KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)]
        if conv.version == "v0"
        else None
    )
        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids,
                images=batch_image_tensor.half().cuda(),
                image_sizes=[image.shape[1:] for image in batch_image_tensor],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                use_cache=True)
        generated_outputs = tokenizer.batch_decode(
                    output_ids[:, batch_input_ids.shape[1] :], skip_special_tokens=True
                )
        generated_outputs = [out.strip() for out in generated_outputs]
        generated_outputs = [
            out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs
        ]
        for question, output in zip(batch, generated_outputs):
            question["pt_output"] = output
            question["pt_answer_list"] = get_answerlist(output)
            ans_file.write(json.dumps(question) + "\n")
            ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)