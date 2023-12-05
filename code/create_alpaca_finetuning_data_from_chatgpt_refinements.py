import argparse
import logging
import re
import json 
from eval_alpaca import eval, get_error_text
from datasets import Dataset, load_dataset, concatenate_datasets
import os 

def format_prompt(text, tests, sample_code):
    # Add code prefix
    fn_name = tests[0]['input'].split("(")[0]
    fn_search = re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", sample_code)
    if fn_search is None:
        return None 
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'"""\n{text}\n"""\n\n{code_prefix}\n'
    return prompt


def load_scored_data(feedback_path):
    d = load_dataset("json", data_files={"train": feedback_path})["train"].map(
        lambda _, idx: {"row_id": idx},
        with_indices=True,
    )
    print(f"Initial length of d: {len(d)}")
    d = d.filter(lambda example: example["passed"])
    print(f"Length of d after filtering for passed: {len(d)}")
    return d


def dedupe_dataset(dataset):
    cols = dataset.column_names
    row_set = set()
    for ex in dataset:
        ex_tuple = tuple(ex[col] for col in cols)
        row_set.add(ex_tuple)
    deduped = {k: [row[i] for row in row_set] for i, k in enumerate(cols)}
    return Dataset.from_dict(deduped)


def remove_prefix_and_func_sig(code, func_sig):
    if f"{func_sig}\r\n" in code:
        return code[code.rfind(f"{func_sig}\r\n") + len(f"{func_sig}\r\n") :]
    elif f"{func_sig} \r\n" in code:
        return code[code.rfind(f"{func_sig} \r\n") + len(f"{func_sig} \r\n") :]
    elif f"{func_sig}\n" in code:
        return code[code.rfind(f"{func_sig}\n") + len(f"{func_sig}\n") :]
    elif f"{func_sig}" in code:
        return code[code.rfind(f"{func_sig}") + len(f"{func_sig}") :]
    else:
        return code

def replace_all(pattern, repl, string) -> str:
    occurences = re.findall(pattern, string, re.IGNORECASE)
    for occurence in occurences:
       string = string.replace(occurence, repl)
    return string
   
def parse_chatgpt_output(completion, orig_completion):
    feedback = ""
    if "RECTIFIED CODE" in completion:
        refinement_str = "RECTIFIED CODE"
        ref_end_idx = completion.rfind(refinement_str) + len(refinement_str)
        feedback = completion[:completion.rfind(refinement_str)].strip("\n")
        if "FEEDBACK" in feedback:
            if "FEEDBACK:" in feedback:
                feedback_str = "FEEDBACK:"
            elif "FEEDBACK" in feedback:
                feedback_str = "FEEDBACK"
            fd_end_idx = feedback.rfind(feedback_str) + len(feedback_str)
            feedback = feedback[fd_end_idx:].strip("\n")
        completion = completion[ref_end_idx:].strip("\n")

    if '```python' in completion:
        completion = completion.split('```')[completion.count('```') - 1]
        if completion.startswith("python"):
            completion = completion[len("python"):]
    elif '```' in completion:
        completion = completion.split('```')[completion.count('```') - 1]
    else:
        code = []
        for line in completion.split("\n"):
            if line.startswith(" ") or line.startswith("\t") or line.strip().startswith("def ") or line.strip().startswith("import ") or line.strip().startswith("from "):
                code.append(line)

        if len(code) <= 1 and orig_completion:
            completion = orig_completion
        else:
            completion = "\n".join(code)
    
    return feedback, completion




def fill_ft_prompt(text, to_replace):
    for k, v in to_replace.items():
        text = text.replace(k, v)
    return text 


def remove_header_from_code(code, prompt):
    func_sig = code.split(":")[0]+":"
    code = code[len(func_sig):]
    prompt = '"""'.join(prompt.split('"""')[:-1])
    prompt = prompt +'"""\n'+ func_sig
    return code, func_sig, prompt 

def create_prompts(args):
    data = load_dataset("ALP/code-alpaca-mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])
    data_task_to_index_dict = {data['task_id'][i]: i for i in range(len(data['task_id']))}
    
    ref_data = [json.loads(x) for x in open(args.refinement_file).readlines()]

    num_chatgpt_pass = 0

    output_gold = args.output_gold_file 
    output_gold_nocg = args.output_gold_nocg_file 
    output_rectification = args.output_refinement_file 
    output_rectification_nofilter = args.output_refinement_nofilter_file 
    output_pseudo_rectification = args.output_gold_refinement_file 
    output_pseudo_rectification_nofilter = args.output_gold_refinement_nofilter_file 

    if not args.no_output_gold_data and (not os.path.exists(output_gold) or not os.path.exists(output_gold_nocg)):

        nocg_ft_data = {
            "finetuning_prompt": [],
            "finetuning_completion": [],
            "task_id": [],
        }
        
        ft_data = {
            "finetuning_prompt": [],
            "finetuning_completion": [],
            "task_id": [],
        }
        for ex in ref_data:
            task_id = ex['task_id']
            prompt = ex['prompt']    
            chatgpt_gold_code = None 
            if "chatgpt_gold" in ex:
                chatgpt_gold_code = ex["chatgpt_gold"]
            
            data_idx = data_task_to_index_dict[task_id]

            text = data['text'][data_idx]
            # Remove method signature prefix
            gold_code = data["code"][data_idx]
            if gold_code is None:
                logging.warning(
                    f"Could not find function signature {func_sig} in gold code.\nGold code:\n{gold_code}"
                )
                continue

            tests = data["test_list"][data_idx]
            # Get the original reformatted MBPP prompt
            orig_prompt = format_prompt(text=text, tests=tests, sample_code=gold_code)

            if chatgpt_gold_code:
                pass_value, _ = eval(completion=chatgpt_gold_code, tests=tests)
                if not pass_value:
                    print (task_id, "did not pass")
                    chatgpt_gold_code = None
                else:
                    num_chatgpt_pass += 1
                chatgpt_gold_code,_, prompt_chatgpt = remove_header_from_code(chatgpt_gold_code, orig_prompt)

 
            gold_code, _, _ = remove_header_from_code(gold_code, orig_prompt)

            
            nocg_ft_data["finetuning_prompt"].append(orig_prompt)
            nocg_ft_data["finetuning_completion"].append(gold_code)
            nocg_ft_data["task_id"].append(task_id)

            ft_data["finetuning_prompt"].append(orig_prompt)
            ft_data["finetuning_completion"].append(gold_code)
            ft_data["task_id"].append(task_id)
        
            if len(nocg_ft_data["task_id"])%10==0:
                print ("finished ", len(nocg_ft_data["task_id"]))
            if chatgpt_gold_code:
                ft_data["finetuning_prompt"].append(prompt_chatgpt)
                ft_data["finetuning_completion"].append(chatgpt_gold_code)
                ft_data["task_id"].append(task_id)


        ft_data = Dataset.from_dict(ft_data)
        nocg_ft_data = Dataset.from_dict(nocg_ft_data)

        if args.sample_size is not None:
            n = min(len(ft_data), args.sample_size)
            ft_data = ft_data.shuffle().select(range(n))

            n = min(len(nocg_ft_data), args.sample_size)
            nocg_ft_data = nocg_ft_data.shuffle().select(range(n))

        ft_data.to_json(output_gold)

        nocg_ft_data.to_json(output_gold_nocg)

    else:
        print ("Already found files:\n",output_gold, '\n', output_gold_nocg)

    rec_ft_data = {
        "finetuning_prompt": [],
        "finetuning_completion": [],
        "task_id": [],
    }

    rec_ft_data_nofilter = {
        "finetuning_prompt": [],
        "finetuning_completion": [],
        "task_id": [],
    }

    pseudo_rec_ft_data_nofilter = {
        "finetuning_prompt": [],
        "finetuning_completion": [],
        "task_id": [],
    }

    pseudo_rec_ft_data = {
        "finetuning_prompt": [],
        "finetuning_completion": [],
        "task_id": [],
    }

    num_passed = 0
    if args.feedback_type == 'nl':
        prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
    elif args.feedback_type == 'auto':
        prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'
    ft_rectify_prompt = open(prompt_filepath).read()

    for i, ex in enumerate(ref_data):
        task_id = ex["task_id"]
        prompt = ex['prompt']  
        data_idx = data_task_to_index_dict[task_id]

        text = data['text'][data_idx]
        # Remove method signature prefix
        gold_code = data["code"][data_idx]
        if gold_code is None:
            logging.warning(
                f"Could not find function signature {func_sig} in gold code.\nGold code:\n{gold_code}"
            )
            continue

        tests = data["test_list"][data_idx]
        prompt = format_prompt(text=text, tests=tests, sample_code=gold_code)
        orig_completion = ex["original_completion"]
        data_idx = data_task_to_index_dict[task_id]
        tests = data["test_list"][data_idx]

        feedback, rectified = parse_chatgpt_output(ex["chatgpt_output"], orig_completion=orig_completion)
        if 'original_error_text' in ex:
            error_text = ex['original_error_text']
        else:
            orig_pass_value, orig_return_value = eval(completion=orig_completion, tests=tests)
            error_text = get_error_text(return_value=orig_return_value)

        
        pass_value, return_value = eval(completion=rectified,  tests=tests)

        gold_code, _, _ = remove_header_from_code(gold_code, orig_prompt)

        prompt_without_head = '"""'.join(prompt.split('"""')[:-1])+'"""'
        rectified, func_sig, _ = remove_header_from_code(rectified, prompt)

        to_replace = {}
        to_replace["<<TASK>>"] = prompt_without_head 
        to_replace["<<CODE>>"] = orig_completion
        to_replace["<<HEADER>>"] = func_sig
        to_replace["<<FEEDBACK>>"] = feedback
        to_replace["<<ERROR>>"] = error_text

        rectify_prompt = fill_ft_prompt(ft_rectify_prompt, to_replace=to_replace)

        pseudo_rec_ft_data_nofilter["task_id"].append(task_id)
        pseudo_rec_ft_data_nofilter["finetuning_prompt"].append(rectify_prompt)
        pseudo_rec_ft_data_nofilter["finetuning_completion"].append(gold_code)

        rec_ft_data_nofilter["task_id"].append(task_id)
        rec_ft_data_nofilter["finetuning_prompt"].append(rectify_prompt)
        rec_ft_data_nofilter["finetuning_completion"].append(rectified)            

        if not pass_value:
            print (task_id, "did not pass")
            continue
        num_passed += 1

        pseudo_rec_ft_data["task_id"].append(task_id)
        pseudo_rec_ft_data["finetuning_prompt"].append(rectify_prompt)
        pseudo_rec_ft_data["finetuning_completion"].append(gold_code)

        rec_ft_data["task_id"].append(task_id)
        rec_ft_data["finetuning_prompt"].append(rectify_prompt)
        rec_ft_data["finetuning_completion"].append(rectified)        

        if len(rec_ft_data["task_id"])%10==0:
            print ("finished ", len(rec_ft_data["task_id"]), " out of ", i)

        #print ("RECTIFY PROMPT: ", rectify_prompt, '\n---')
        #print (rectified, "\n========================================\n\n")

    print ("% passed : ", 100.0*num_passed/float(len(ref_data)))
    print ("% passed (chatgpt independent) ", 100.0*num_chatgpt_pass/float(len(ref_data)))
    rec_ft_data = Dataset.from_dict(rec_ft_data)
    rec_ft_data_nofilter = Dataset.from_dict(rec_ft_data_nofilter)
    pseudo_rec_ft_data = Dataset.from_dict(pseudo_rec_ft_data)
    pseudo_rec_ft_data_nofilter = Dataset.from_dict(pseudo_rec_ft_data_nofilter)


    rec_ft_data.to_json(output_rectification)
    rec_ft_data_nofilter.to_json(output_rectification_nofilter)
    pseudo_rec_ft_data.to_json(output_pseudo_rectification)
    pseudo_rec_ft_data_nofilter.to_json(output_pseudo_rectification_nofilter)


def parse_args(input_args):
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning prompts from model-generated refinements. Also generate FT prompts for those same task IDs from the original MBPP dataset using gold code."
    )
    parser.add_argument(
        "--refinement-file",
        type=str,
        help="Path to file containing evaluated refinements from ChatGPT. Needs to have the following columns: passed, task_id, prompt, completion, chatgpt_feedback",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to output data files in."
    )
    parser.add_argument(
        "--no-output-gold-data",
        action="store_true",
        help="If set, will not output finetuning files for gold completions.",
    )
    
    parser.add_argument(
        "--feedback-type",
        type=str,
        help="type of feedback needed: nl or auto"
    )

    parser.add_argument(
        "output-gold-file",
        type=str, 
        help="Path to output file containing chatgpt gold data "
    )

    parser.add_argument(
        "output-gold-nocg-file",
        type=str,
        help="Path to output file containing nonchatgpt gold data. In case of Code-Alpaca, since no such data exists, we dump chatgpt gold data here"
    )

    parser.add_argument(
        "--output-refinement-file",
        type=str, 
        help="Path to output file containing chatgpt refinement data"
    )

    parser.add_argument(
        "--output-refinement-nofilter-file",
        type=str, 
        help="Path to output file containing chatgpt refinement data without any eval based filtering"
    )

    parser.add_argument(
        "output-gold-refinement-file",
        type=str,
        help="Path to output file containing pseudo refinement data (with gold as rectified code)"
    )

    parser.add_argument(
        "output-gold-refinement-nofilter-file",
        type=str, 
        help="Path to output file containing pseudo refinement data for all instances where the original model had produced incorrect results"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args(None)
    create_prompts(args)


if __name__ == "__main__":
    main()