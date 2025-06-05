import json
import random
import os
import re
import tqdm
import argparse
from collections import defaultdict
from sacrebleu.metrics import BLEU
from metric import TEDS


def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str

def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip() 
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def convert_latex_table_to_html(latex_table):
    """
    Converts a markdown table to html string for TEDS computation.
    In the MMTab-eval, we only consider latex tables with similar structures of markdown tables.
    For other latex tables with compicated structures like merged cells, you need to rewrite this function to convert them.
    """
    # remove extra code block tokens like '```latex' and '```
    latex_table = latex_table.strip('```latex').strip('```').strip() 
    latex_table = latex_table.replace('\n', ' ')
    row_str_list = [row_str.strip('\n').strip('\\') for row_str in latex_table.split('\hline')[1:-1]]
    table_rows = []
    for row_str in row_str_list:
        one_row = []
        for c in row_str.split('&'):
            if set(c) != set(' '):
                one_row.append(c.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table

# Read inference results of LLaVA model (merged.jsonl)
def read_llava_prediction_file(file_path):
    """
    Read LLaVA's inference results (e.g., merge.jsonl) and extract data of different benchmarks based on 'category' field.
    """
    predict_results = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            item = json.loads(line.strip())
            predict_results.append(item)
    print("Predicted Sample Number:",len(predict_results))
    benchmark_name_to_predicted_item_list = defaultdict(list)
    for item in predict_results:
        if "question_id" not in item:
            item["question_id"] = item["item_id"]
        if "text" not in item:
            item["text"] = item["pt_output"]
        item_id = item['question_id']
        category = item.get('category', "") # {dataset_name}_for_{task_name}, e.g., TabFact_for_TFV
        if category:
            dataset_name = category.split('_for_')[0] # e.g., TabFact
            task_name = category.split('_for_')[1] # e.g., TFV
        else:
            dataset_name, task_name = item["dataset_name"], item["task_type"]
        # for table structure understanding tasks, benchmark name is the task name
        if task_name not in ['TSD','TCL','RCE','MCD','TCE','TR','OOD_TSD','OOD_TCL','OOD_RCE','OOD_TCE']:
            benchmark_name = dataset_name
        else:
            benchmark_name = task_name
        benchmark_name_to_predicted_item_list[benchmark_name].append(item)
    for benchmark_name,  predicted_item_list in benchmark_name_to_predicted_item_list.items():
        item_num = len(predicted_item_list)
        print(f'benchmark name: {benchmark_name}, test data num: {item_num}')
    return benchmark_name_to_predicted_item_list


def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub('[\"\']+',"\"",answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = [model_output]
        return predicted_answer
    else:
        return []

def evaluate_tqa_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table question answering (TQA) and table fact verification (TFV) benchmark.
    Metric: accuracy.
    Note that some baseline models can not strictly follow instructions to output the final answer in the required JSON format.
    For instance, Qwen-VL may only output a short answer due to the potential overfitting of training data.
    In such cases, the evaluation script needs to be changed according to the characteristic of certain model output.
    """
    correct_item_list = []
    wrong_item_list = []
    failed_item_list = []
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            # parse the predicted answer list
            predicted_answer_list = extract_tqa_answer_list(model_output)
            gold_answer_list = ori_item['answer_list']
            # Sometimes the order of multiple answer text is not necessarily same as the gold answer,
            # so we convert the answer list to a set for comparison
            if set(gold_answer_list) == set(predicted_answer_list):
                correct_item_list.append(item)
            else:
                # if benchmark_name == "WTQ":
                #     print(gold_answer_list, predicted_answer_list)
                wrong_item_list.append(item)
        except Exception:
            failed_item_list.append(item)
            
    print("Benchmark: ",benchmark_name)
    correct_num = len(correct_item_list)
    total_sample_num = len(pred_item_list)
    print("Accuracy: ", correct_num/total_sample_num)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)


def extract_tfv_answer_list(model_output):
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub('[\"\']+',"\"",answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = [model_output]
        return [answer.split(" ")[0] if "with" in answer else answer for answer in predicted_answer]
    else:
        return []


def evaluate_tfv_questions(benchmark_name, pred_item_list):
    correct_item_list = []
    wrong_item_list = []
    failed_item_list = []
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            # parse the predicted answer list
            predicted_answer_list = extract_tfv_answer_list(model_output)
            gold_answer_list = ori_item['answer_list']
            # Sometimes the order of multiple answer text is not necessarily same as the gold answer,
            # so we convert the answer list to a set for comparison
            if set(gold_answer_list) == set(predicted_answer_list):
                correct_item_list.append(item)
            else:
                wrong_item_list.append(item)
                # print(gold_answer_list, predicted_answer_list)
        except Exception:
            failed_item_list.append(item)
            
    print("Benchmark: ",benchmark_name)
    correct_num = len(correct_item_list)
    total_sample_num = len(pred_item_list)
    print("Accuracy: ", correct_num/total_sample_num)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_tabmcq_questions(benchmark_name,pred_item_list):
    """
    Evaluation for TabMCQ benchmark.
    Metric: accuracy. 
    """
    correct_item_list = []
    wrong_item_list = []
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace(']','')
            model_output = model_output.replace('[','')
            ret = re.match('.*({[\"\']answer[\"\']\:\s?.*?}).*',model_output)
            # parse predicted answer
            if ret is not None:
                answer_str = ret.group(1)
                answer_item = eval(answer_str)
                predicted_answer = answer_item['answer']
                if type(predicted_answer) == list:
                    predicted_answer = predicted_answer[0]
                gold_answer_str = ori_item['answer_list'][0] # e.g., '(D) Blood'
                # Sometimes the predicted answer does not contain option letter like '(D)'
                # To deal with such cases, we also consider removing the option letter in ground truth for comparison
                if predicted_answer == gold_answer_str or predicted_answer == ' '.join(gold_answer_str.split(' ')[1:]):
                    correct_item_list.append(item)
                else:
                    wrong_item_list.append(item)
            else:
                failed_item_list.append(item)
        except Exception:
            failed_item_list.append(item)
            
    print(f"Benchmark: {benchmark_name}")
    total_sample_num = len(pred_item_list)
    correct_num = len(correct_item_list)
    print("Accuracy: ",correct_num/total_sample_num)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_text_generation_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table-to-text benchmark.
    Metric: bleu.
    More metrics like ROUGE or LLM-as-a-judge rating are needed for a more robust evaluation.
    """
    bleu = BLEU()
    output_text_list = [] # output text 
    reference_text_list = [] # reference text list
    for item in pred_item_list:
        pred_text = item['text']
        item_id = item['question_id']
        ori_item = item_id_to_test_item[item_id]
        gold_text = ori_item['output']
        assert gold_text not in ['','None']
        output_text_list.append(pred_text)
        reference_text_list.append(gold_text)
    assert len(output_text_list) == len(reference_text_list)
    bleu_score = bleu.corpus_score(output_text_list, [reference_text_list])
    print("Benchmark: ",benchmark_name)
    print("BLEU score:",bleu_score)
    print("-"*20)

teds = TEDS(n_jobs=36)
def evaluate_tr_questions(pred_item_list):
    """
    Evaluation for table recognition (TR) benchmark.
    Metric: TEDS score (Tree-Edit-Distance-based Similarity)
    We directly use the TEDS object from the PubTabNet code to compute TEDS score based on html table string.
    For test samples with latex and markdown table format, we convert them into html table string to compute TEDS score.
    Source of TEDS computation code: https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src
    """
    html_item_list = []
    markdown_item_list = []
    latex_item_list = []
    item_id_to_pred = {}
    item_id_to_gold = {}
    failed_item_list = [] # store items that we failed to extract or build html table string
    
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            query_type = ori_item['original_query_type'] # e.g., 'table recognition with HTML', 'table recognition with Latex'
            item['query_type'] = query_type
            pred_table_repr = item['text']
            gold_table_repr = ori_item['output']
            # seperate different table recognition questions based on 'query_type' field
            if 'HTML' in query_type:
                # extract html str from the output using regular expression
                pred_table_repr = pred_table_repr.replace('\n','')
                if "<body" in pred_table_repr:
                    pred_table_repr = re.findall('<body.*',pred_table_repr)[0]
                elif "<table" in pred_table_repr:
                    pred_table_repr = re.findall('<table.*',pred_table_repr)[0]
                else:
                    failed_item_list.append(item)
                    continue
                pred_html_table = wrap_html_table(pred_table_repr)
                gold_html_table = wrap_html_table(gold_table_repr)
            elif 'Latex' in query_type:
                # convert latex table string to html table for TEDS computation
                # note: for now, this conversion only support flat table with the first row as column headers.
                pred_html_table = convert_latex_table_to_html(pred_table_repr)
                gold_html_table = convert_latex_table_to_html(gold_table_repr)
            else:
                # convert markdown table string to html table for TEDS computation
                pred_html_table = convert_markdown_table_to_html(pred_table_repr)
                gold_html_table = convert_markdown_table_to_html(gold_table_repr)
            item['pred_html_table'] = pred_html_table
            item['gold_html_table'] = gold_html_table
            item_id_to_pred[item_id] = pred_html_table
            item_id_to_gold[item_id] = gold_html_table
        except Exception as e:
            item['exception'] = e
            failed_item_list.append(item)
    # teds.batch_evaluate() returns a dictionary of {item_id: TEDS score}
    item_id_to_teds_score = teds.batch_evaluate(item_id_to_pred, item_id_to_gold)
    item_list_without_teds_score = [] 
    for item in pred_item_list:
        item_id = item['question_id']
        query_type = item['query_type']
        if item_id in item_id_to_teds_score:
            TEDS = item_id_to_teds_score[item_id]
            if type(TEDS) != float:
                item_list_without_teds_score.append(item)
                continue
            item['TEDS'] = TEDS
            if 'HTML' in query_type:
                html_item_list.append(item)
            elif 'Latex' in query_type:
                latex_item_list.append(item)
            else:
                markdown_item_list.append(item)
        else:
            item_list_without_teds_score.append(item)  
    print("Benchmark: TR")   
    print("HTML sample number:",len(html_item_list))
    print("Markdown sample number:",len(markdown_item_list))
    print("Latex sample number:",len(latex_item_list))
    print("")
    if len(html_item_list) != 0:
        html_ave_teds_score = sum([item['TEDS'] for item in html_item_list])/len(html_item_list)
        print("Average TEDS score for HTML samples:",html_ave_teds_score)
    if len(markdown_item_list) != 0:
        markdown_ave_teds_score = sum([item['TEDS'] for item in markdown_item_list])/len(markdown_item_list)
        print("Average TEDS score for Markdown samples:",markdown_ave_teds_score)
    if len(latex_item_list) != 0:
        latex_ave_teds_score = sum([item['TEDS'] for item in latex_item_list])/len(latex_item_list)
        print("Average TEDS score for Latex samples:",latex_ave_teds_score)
        print("")
        
    total_sample_num = len(pred_item_list)
    sample_num_without_teds_score = len(item_list_without_teds_score)
    sample_num_without_html_table_string = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {sample_num_without_html_table_string} samples that we can not extract HTML table string.")
    print(f"There are {sample_num_without_teds_score} samples that we can not compute TEDS score.")
    print("-"*20)

def evaluate_mcd_questions(benchmark_name,pred_item_list):
    """
    Evaluation for merged cell detection (MCD) benchmark.
    Metric: precision, recall and F1 score
    """
    pred_cell_num = 0 # number of predicted merged cells 
    gold_cell_num = 0 # number of gold merged cells
    correct_cell_num = 0 # number of predicted merged cells which are correct
    failed_item_list = []
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            model_output = model_output.replace('\n',' ')
            gold_answer_list = ori_item['answer_list']
            merged_cell_region_list = []
            gold_cell_num += len(gold_answer_list)
            if gold_answer_list == ['None']: # There are no merged cells in the table
                # Different models may use different sentences to express 'there is no merged cell'.
                # In such cases, you need to include more expressions for a more accurate evaluation.
                if "does not contain any merged cells" in model_output.lower() or "no merged cell" in model_output.lower():
                    correct_cell_num += 1
                    pred_cell_num += 1
                else:
                    pred_cell_num += 1
            else:  # There are merged cells in the table
                # parse the ground truth coordinates of merged cells
                for answer_str in gold_answer_list:
                    gold_answer_item = eval(answer_str)
                    top_row_id, left_col_id = gold_answer_item['top-left']
                    bottom_row_id, right_col_id = gold_answer_item['bottom-right']
                    gold_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
                    merged_cell_region_list.append(gold_merged_region_repr)
                # parse the predicted coordinates of merged cells
                pred_answer_str_list = re.findall('{[\"\']top-left[\"\']\:.*?,\s?[\"\']bottom-right[\"\']\:.*?}',model_output)
                for answer_str in pred_answer_str_list:
                    pred_answer_item = eval(answer_str)
                    top_row_id, left_col_id = pred_answer_item['top-left']
                    bottom_row_id, right_col_id = pred_answer_item['bottom-right']
                    pred_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
                    if pred_merged_region_repr in merged_cell_region_list:
                        correct_cell_num += 1
                pred_cell_num += len(pred_answer_str_list)
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
             
    print(f"Benchmark: {benchmark_name}")
    P = correct_cell_num / pred_cell_num
    R = correct_cell_num / gold_cell_num
    print("Precision:",P)
    print("Recall:",R)
    if P+R == 0:
        F1 = 0
    else:
        F1 = 2*P*R/(P+R)
    print("F1 score:",F1)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_tcl_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table cell locating (TCL) benchmark.
    Metric: cell-level accuracy
    """
    total_cell_num = 0
    correct_cell_num = 0
    failed_item_list = []
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace('\\','')
            gold_output = ori_item['output']
            # parse the ground truth cell locations (row_id, column_id)
            # example gold_dict_str_list = [('Raúl Hidalgo', '(13, 1)'),
            #                               ('Year', 'DOES NOT EXIST')],
            gold_dict_str_list = re.findall('{[\"\']value[\"\']\:\s?[\"\'](.*?)[\"\'],\s?[\"\']location[\"\']\:\s?[\"\']?(.*?)[\"\']?}',gold_output)
            cell_str_to_location = {}
            for cell_str,location_str in gold_dict_str_list:
                cell_str_to_location[cell_str] = location_str
            total_cell_num += len(cell_str_to_location)
            item['cell_str_to_gold_location'] = cell_str_to_location
            # parse the predicted cell locations
            pred_dict_str_list = re.findall('{[\"\']value[\"\']\:\s?[\"\'](.*?)[\"\'],\s?[\"\']location[\"\']\:\s?[\"\']?(.*?)[\"\']?}',model_output)
            cell_str_to_pred_location = {}
            for cell_str,location_str in pred_dict_str_list:
                if cell_str in cell_str_to_location:
                    gold_cell_location = cell_str_to_location[cell_str]
                    pred_cell_location = location_str
                    cell_str_to_pred_location[cell_str] = location_str
                    if str(gold_cell_location).lower() == str(pred_cell_location).lower():
                        correct_cell_num += 1 
            item['cell_str_to_pred_location'] = cell_str_to_pred_location
                
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    print("Cell-level accuracy:",correct_cell_num/total_cell_num)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    
def evaluate_rce_questions(benchmark_name,pred_item_list):
    """
    Evaluation for row and column extraction (RCE) benchmark.
    Metric: row and column level F1 score
    """
    row_correct_cell_num = 0 
    row_pred_cell_num = 0 
    row_ori_cell_num = 0 
    column_correct_cell_num = 0 
    column_pred_cell_num = 0 
    column_ori_cell_num = 0
    
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            image_id = ori_item['image_id']
            table_rows = table_id_to_test_table[image_id]['table_rows']
            model_output = item['text']
            
            row_id_to_correct_cell_list = {}
            col_id_to_correct_cell_list = {}
            # parse the predicted cells of specific row or column
            match_group_tuple_list = re.findall('{[\"\']row_id[\"\']\:(.*?),\s?[\"\']cell_list[\"\']\:(.*?)}|{[\"\']column_id[\"\']\:(.*?),\s?[\"\']cell_list[\"\']\:(.*?)}',model_output)
            for matched_tuple in match_group_tuple_list:
                if matched_tuple[0] != '': # extract cells from a specific row
                    row_id = int(eval(matched_tuple[0]))
                    pred_cell_list = eval(matched_tuple[1])
                    target_cell_list = table_rows[row_id-1] # the ground truth cells in the original row
                    row_pred_cell_num += len(pred_cell_list)
                    row_ori_cell_num += len(target_cell_list)
                    correct_cell_list = [c for c in pred_cell_list if c in target_cell_list] # predicted cells that are also in the ground truth
                    row_correct_cell_num += len(correct_cell_list)
                    row_id_to_correct_cell_list[row_id] = correct_cell_list
                else: # extract cells from a specific column
                    column_id = int(eval(matched_tuple[2]))
                    pred_cell_list = eval(matched_tuple[3])
                    target_cell_list = [] # the ground truth cells in the original column  
                    for row in table_rows:
                        if len(row) == 1:
                            target_cell_list.append(row[0])
                        else:
                            target_cell_list.append(row[column_id-1])
                    column_pred_cell_num += len(pred_cell_list)
                    column_ori_cell_num += len(target_cell_list)
                    correct_cell_list = [c for c in pred_cell_list if c in target_cell_list] # predicted cells that are also in the ground truth
                    column_correct_cell_num += len(correct_cell_list)
                    col_id_to_correct_cell_list[column_id] = correct_cell_list
            item['row_id_to_correct_cell_list'] = row_id_to_correct_cell_list
            item['col_id_to_correct_cell_list'] = col_id_to_correct_cell_list
            
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    row_P = row_correct_cell_num/row_pred_cell_num # row-level precision
    row_R = row_correct_cell_num/row_ori_cell_num # row-level recall
    row_F1 = 2*row_P*row_R/(row_P+row_R) # row-level F1
    col_P = column_correct_cell_num/column_pred_cell_num # column-level precision
    col_R = column_correct_cell_num/column_ori_cell_num # column-level recall
    col_F1 = 2*col_P*col_R/(col_P+col_R) # column-level F1
    
    print("Row-level Precision:",row_P)
    print("Row-level Recall:",row_R)
    print("Row-level F1:",row_F1)
    print("")
    print("Column-level Precision:",col_P)
    print("Column-level Recall:",col_R)
    print("Column-level F1:",col_F1)
    
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_tce_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table cell extraction (TCE) benchmark.
    Metric: cell-level accuracy
    """
    total_cell_num = 0
    correct_cell_num = 0
    failed_item_list = []
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text']
            model_output = model_output.replace('\n',' ')
            gold_output = ori_item['output']
            # parse ground truth cell value
            gold_dict_str_list = re.findall('{[\"\']row_id[\"\']\:.*?[\"\']column_id[\"\']\:.*?[\"\']cell_value[\"\']\:.*?}',gold_output)
            cell_location_to_cell_str = {}
            for dict_str in gold_dict_str_list:
                cell_item = eval(dict_str)
                row_id = cell_item['row_id']
                col_id = cell_item['column_id']
                gold_cell_value = cell_item['cell_value']
                cell_location_to_cell_str[f"{row_id}_{col_id}"] = gold_cell_value
            total_cell_num += len(cell_location_to_cell_str)
            # parse predicted cell value
            pred_dict_str_list = re.findall('{[\"\']row_id[\"\']\:.*?[\"\']column_id[\"\']\:.*?[\"\']cell_value[\"\']\:.*?}',model_output)
            cell_location_to_pred_str = {}
            for dict_str in pred_dict_str_list:
                # some output may contain extra '[' or ']'
                dict_str = dict_str.replace(']','')
                dict_str = dict_str.replace('[','')
                cell_item = eval(dict_str)
                row_id = cell_item['row_id']
                col_id = cell_item['column_id']
                cell_location = f"{row_id}_{col_id}"
                if (cell_location in cell_location_to_cell_str) and (cell_location not in cell_location_to_pred_str) :
                    gold_cell_value = cell_location_to_cell_str[cell_location]
                    pred_cell_value = cell_item['cell_value']
                    cell_location_to_pred_str[cell_location] = pred_cell_value
                    if str(pred_cell_value).lower() == str(gold_cell_value).lower():
                        correct_cell_num += 1
    
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    print("cell level accuracy: ",correct_cell_num/total_cell_num)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_tsd_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table size detection (TSD) benchmark.
    Metric: row number and column number accuracy
    """
    row_correct_num = 0
    col_correct_num = 0
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            item_id = item['question_id']
            ori_item = item_id_to_test_item[item_id]
            model_output = item['text'].lower()
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace('\\','')
            # parse predicted row number and column number
            if 'row_number' in model_output:
                ret = re.match('.*({.*[\"\']row_number[\"\']:.*[\"\']column_number[\"\']:.*}).*',model_output)
                answer_str = ret.group(1)
                answer_item = eval(answer_str)
                pred_row_number = answer_item['row_number']
                pred_col_number = answer_item['column_number']
            else:
                ret = re.match('.*(\d+) rows and (\d+) columns.*',model_output)
                pred_row_number = ret.group(1)
                pred_col_number = ret.group(2)
            # extract ground truth row number and column number
            gold_answer_tuple = ori_item['answer_list'][0]
            gold_row_number = str(gold_answer_tuple[0])
            gold_col_number = str(gold_answer_tuple[1])
            if pred_row_number == gold_row_number:
                row_correct_num += 1
            if pred_col_number == gold_col_number:
                col_correct_num += 1

        except Exception as e:
            item['exception'] = e
            failed_item_list.append(item)
            
    print(f"Benchmark: {benchmark_name}")
    total_sample_num = len(pred_item_list)
    print("row number accuracy:",row_correct_num/total_sample_num)
    print("column number accuracy:",col_correct_num/total_sample_num)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="/home/zbb/code/LLaVA/eval_results/answers/MMTab_eval/table-llava-v1.5-7b-clipvit/merge.jsonl")
    parser.add_argument('--test_data', default="/home/zbb/data/Table-LlaVA/LLaVA-Inference/MMTab-eval_test_data_49K.json")
    parser.add_argument('--test_table', default="/home/zbb/data/Table-LlaVA/LLaVA-Inference/MMTab-eval_test_tables_23K.json")
    args, unknown = parser.parse_known_args()

    input_file = args.input_file
    test_data = args.test_data
    benchmark_name_to_predicted_item_list = read_llava_prediction_file(input_file)
    # read the ground truth data
    MMTab_eval_test_data = json.load(open(test_data))
    # item_id --> test data
    item_id_to_test_item = {}
    for item in MMTab_eval_test_data:
        item_id = item['item_id']
        item_id_to_test_item[item_id] = item
    print("MMTab-eval data num: ",len(MMTab_eval_test_data))
    # table_id --> test table
    test_tables = args.test_table
    MMTab_eval_test_tables = json.load(open(test_tables))
    table_id_to_test_table = {}
    for table_item in MMTab_eval_test_tables:
        table_id = table_item['image_id']
        table_id_to_test_table[table_id] = table_item
    print("MMTab-eval table num: ",len(table_id_to_test_table))

    benchmark_name_list = ['TSD','OOD_TSD','TCE','OOD_TCE','TCL','OOD_TCL','MCD','RCE','OOD_RCE','TABMWP','WTQ','HiTab','TAT-QA',
                      'FeTaQA','AIT-QA','TabMCQ','TabFact','InfoTabs','PubHealthTab',
                      'HiTab_t2t','Rotowire','WikiBIO']
    for benchmark_name in benchmark_name_list:
        predicted_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
        if benchmark_name in ['TSD','OOD_TSD']:
            evaluate_tsd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCE','OOD_TCE']:
            evaluate_tce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCL','OOD_TCL']:
            evaluate_tcl_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['RCE','OOD_RCE']:
            evaluate_rce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'MCD':
            evaluate_mcd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'TabMCQ':
            evaluate_tabmcq_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['FeTaQA','HiTab_t2t','Rotowire','WikiBIO']:
            evaluate_text_generation_questions(benchmark_name,predicted_item_list)
        else:
            evaluate_tqa_questions(benchmark_name,predicted_item_list)

        # for benchmark_name in benchmark_name_list:
        #     predicted_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
        #     if benchmark_name == 'TabMCQ':
        #         evaluate_tabmcq_questions(benchmark_name,predicted_item_list)
        #     elif benchmark_name in ['TabFact','InfoTabs','PubHealthTab']:
        #         evaluate_tfv_questions(benchmark_name, predicted_item_list)
        #     elif benchmark_name in ['FeTaQA','HiTab_t2t','Rotowire','WikiBIO']:
        #         evaluate_text_generation_questions(benchmark_name,predicted_item_list)
        #     else:
        #         evaluate_tqa_questions(benchmark_name,predicted_item_list)
    
    # The evaluation of table recognition task takes about 2.5 minutes.
    benchmark_name = 'TR'
    predicted_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
    evaluate_tr_questions(predicted_item_list)

    benchmark_name = 'ToTTo'
    pred_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
    # Sort the output samples
    totto_sample_index_to_item = {}
    for item in pred_item_list:
        item_id = item['question_id']
        sample_index = int(item_id.strip('ToTTo_test_item_'))
        totto_sample_index_to_item[sample_index] = item
    # Write the output result to a txt file. Each line represents a output text for a sample.
    # The sample order must be the same as the original ToTTo test set.
    sorted_totto_item_list = []
    base_dir = "/".join(input_file.split("/")[:-1])
    # Use this txt file for leaderboard submission
    with open(f'{base_dir}/ToTTo_test_results.txt','w',encoding='utf-8') as f:
        for sample_index in range(len(pred_item_list)):
            item = totto_sample_index_to_item[sample_index]
            output_text = item['text'].replace('\n',' ').replace('\t',' ')
            f.write(output_text+'\n')
            sorted_totto_item_list.append(item)
    print("ToTT test sample number:",len(sorted_totto_item_list))

