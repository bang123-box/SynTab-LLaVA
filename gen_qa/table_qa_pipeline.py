import os
import re
from openai import AzureOpenAI, OpenAI
from datetime import datetime
import json
import tiktoken
import math
import argparse
import re
import random

from table_task_mkprompt import *
from table_task_htmlprompt import *
#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 1.0.0 or higher.

characteristics_list = [ "Numeral Calculations", 
                        "Complex Calculations", 
                        "Tabel Fact Verification", 
                        "Multiple Choice",     
                        "Free Table Question Answering",  
                        "Table Summary"     
                        ]

description_map = {
  "Numeral Calculations": " Numerical Calculations involve:\nAddition: Summing numerical values.\nSubtraction: Finding the difference between numerical values.\nMultiplication: Calculating the product of numerical values.\nDivision: Determining the quotient of numerical values.\nSimple Combinations: Performing basic operations involving combinations of addition, subtraction, multiplication, and division.", #数学运算(加减乘除)

  "Complex Calculations": "Complex Calculations involve operations that combine multiple tasks, including: Data Retrieval, Sorting, Filtering, Mathematical Operations, and so on. Complex calculations integrate these tasks to perform more intricate data analysis.", #复杂数学运算
  
  "Tabel Fact Verification": "Table Fact Verification involves checking the accuracy and validity of information presented in a table against known facts or expected results. Key aspects of table fact verification include:Cross-Checking, Consistency Check, Error Detection, Logical Verification, and so on.", #表格事实验证

  "Multiple Choice": "Multiple Choice in the context of table question answering involves selecting the correct answer from a list of given options based on the data presented in the table. ", #选择题

  "Free Table Question Answering": "The question should require retrieval and reasoning over multiple sources of information in the table,and the answer should integrate both facts and inferences into a coherent sentence that answers the question.", #自由问答

  "Table Summary": "Table Summary involves providing a concise and coherent description of the key information presented in a table.", #Table summary 
}

char_map = {
  "Numeral Calculations": "Calculations_exmaples", 
  "Complex Calculations": "Composition_examples", 
  "Tabel Fact Verification": "FactVerfied_examples", 
  "Multiple Choice": "Multiple_choice_examples", 
  "Free Table Question Answering": "Freetqa_examples", 
  "Table Summary": "Caption_examples", 
}


PROMPT_COT_TEMP = """
You are an AI visual assistant, skilled in analyzing tables. You receive the table in {input_format} format. You should design question-answer pairs between you and a person asking about this table.
Ask diverse questions and give corresponding answers. Only include questions that have definite answers:(1) one can see in the table that the question asks about and can answer confidently;(2) one can determine confidently from the table that it is not in the table.
Do not ask any question that cannot be answered confidently. 

The number of questions needs to be {num}. Include questions asking about {Characteristics}.

All the question and answer pairs should satisfy: 
1. The answers should be in a tone that a visual AI assistant is seeing the table figure and answering the questions.. 
2. The questions should contain at least one reasoning question.
3. Each question-answer pair should contain QUESTION, DETAIL_ANSWER and SHORT_ANSWER. The DETAIL_ANSWER and SHORT_ANSWER must have no logical confilict.
4. For the DETAIL_ANSWER, the answers should be in as much details as possible and should be complemented in the "chain-of-thought" pattern. For the SHORT_ANSWER, the answers should be one word or phase.
5. The format should strictly follow: QUESTION: XXXX\nDETAIL_ANSWER: XXXXX\nSHORT_ANSWER: XXXX\n\nQUESTION: XXXX\nDETAIL_ANSWER: XXXXX\nSHORT_ANSWER: XXXX...

#Here are some examples and remember to follow their format: 
{instruction_examples}
#The input table is: 
{table}

Please follow the given example format strictly, Output:
"""

def contruct_ins(characteristics_list, input_format):
    characteristics_list = [ "Numeral Calculations", 
                        "Complex Calculations",
                        "Tabel Fact Verification", 
                        "Multiple Choice",      
                        "Free Table Question Answering",  
                        "Table Summary"      
                        ]
    M = random.randint(1,3)
    characteristics_sample = random.sample(characteristics_list, M)

    Characteristics = "\n"
    for ch in characteristics_sample:
      Characteristics += ch + ": "
      Characteristics += description_map[ch] + "\n\n"
    Characteristics = Characteristics[:-1]
    instruction_examples = ""
    for ch in characteristics_sample:
      instruction_examples += f"{ch} exmaples:\n"
      examples_str = char_map[ch]
      examples_list = eval(input_format + "_" + examples_str)
      examples = random.choice(examples_list)
      instruction_examples += examples
      instruction_examples += '\n'

    return Characteristics, instruction_examples, ", ".join(characteristics_sample)

def split_list(lst, n):
    # Split a list into n (roughly) equal-sized chunks
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def write_file(args, gpt_answer_list, output_file_name, start,end, gpt_answer_cnt,cnt):
    with open(output_file_name, "a") as f:
        for q in gpt_answer_list:
            f.write(json.dumps(q) + "\n")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time}: {args.num_chunks}_{args.chunk_idx} {start} / {end}, use gpt {gpt_answer_cnt} / {cnt} times")

def construct_qa(args):
    client = OpenAI(
    api_key = "your own keys",     # base_url = "http://0.0.0.0:8000/v1"
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)
    
    questions = json.load(open(os.path.expanduser(args.input_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    output_file_name = f"{args.output_dir}/{args.num_chunks}_{args.chunk_idx}.jsonl"
    if not os.path.exists(output_file_name):
        start = 0
    else:
       with open(output_file_name, "r") as f:
           start = len([line for line in f.readlines()])
        
    end = len(questions)
    print(f"{args.num_chunks}_{args.chunk_idx} {start} / {end}")

    gpt_answer_cnt = 0
    cnt = 0
    gpt_answer_list = []
    while start < end:
        if len(gpt_answer_list) != 0 and len(gpt_answer_list) % 5 == 0:
            write_file(args, gpt_answer_list, output_file_name, start,end, gpt_answer_cnt, cnt)
            gpt_answer_list = []
            
        item = questions[start]
        table = item["conversations"][1]["value"]
        del item["conversations"]

        if table[:6] == "<table":
            input_format = "html"
        else:
            input_format = "markdown"
        try:
            Characteristics, instruction_examples, instruct_types = contruct_ins(characteristics_list, input_format)
            user_prompt = PROMPT_COT_TEMP.format(input_format=input_format, num=1, 
                                                instruction_examples=instruction_examples, 
                                                Characteristics=Characteristics, table=table)
            completion = client.chat.completions.create(
            model= "your model name", # model = "gpt-4-32k"
            messages = [
                  {  
                    "role": "user",
                    "content": f"{user_prompt}",
                  },
            ],
            temperature=0.8,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
            response = completion.choices[0].message.content
            item["response"] = response
            item["prompt"] = f"{user_prompt}"
            item["instruct_types"] = instruct_types
            gpt_answer_cnt += 1
            if gpt_answer_cnt % 20 == 0:
                print(completion)
        except Exception as e:
          print(e)

        start += 1
        cnt += 1
        gpt_answer_list.append(item)

    write_file(args, gpt_answer_list, output_file_name, start,end, gpt_answer_cnt, cnt)

if __name__ == '__main__':
  argparse = argparse.ArgumentParser()
  argparse.add_argument('--input_file', type=str, default='/data/Table/pretrain/pretrain_636179.json')
  argparse.add_argument('--output_dir', type=str, default='/data/Table/pretrain/gen_qa')
  argparse.add_argument("--num_chunks", type=int, default=1)
  argparse.add_argument("--chunk_idx", type=int, default=0)
  args = argparse.parse_args()
  construct_qa(args)


