import argparse
import logging
import re
import json 
from eval_mbpp import eval, get_error_text
from datasets import Dataset, load_dataset, concatenate_datasets

def format_prompt(data, task_id):
    idx = data["task_id"].index(task_id)
    text = data["text"][idx]
    tests = data["test_list"][idx]
    sample_code = data["code"][idx]

    # Create prompt from scratch
    prompt = f'"""\n{text}\n\n'
    # Add the first unit test as an input-output example
    example = tests[0].split("assert ")[-1].replace("==", "=")
    prompt += f">>> Example: {example}\n"

    # Add code prefix
    fn_name = tests[0].split("assert ")[-1].split("(")[0]
    fn_search = re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", sample_code)
    if fn_search is None:
        raise ValueError(
            f"Could not find 'def {fn_name}\(.*\):' in code for task {task_id}."
        )
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'{prompt}"""\n\n{code_prefix}\n'
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
        if "FEEDBACK:" in feedback:
             feedback_str = "FEEDBACK:"
        elif "FEEDBACK" in feedback:
             feedback_str = "FEEDBACK"
        fd_end_idx = feedback.rfind(feedback_str) + len(feedback_str)
        feedback = feedback[fd_end_idx:].strip("\n")
        feedback = replace_all("student's ", "", feedback)
        feedback = replace_all("student", "", feedback)
        completion = completion[ref_end_idx:].strip("\n")

    if '```python' in completion:
        completion = completion.split('```')[1]
        completion = completion[len("python"):]
    elif '```' in completion:
        completion = completion.split('```')[1]
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
    func_sig_pattern = re.search("def .*\n", code)
    func_sig = code[:func_sig_pattern.start()]
    try:
        code = code[func_sig_pattern.end():]
    except:
        code = None 
    #func_sig = code.split(":")[0]+":"
    #code = code[len(func_sig):]
    prompt = '"""'.join(prompt.split('"""')[:-1])
    prompt = prompt +'"""\n'+ func_sig
    return code, func_sig, prompt 

def create_prompts(args):
    data = load_dataset("mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])

    if args.feedback_type == 'nl':
        prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
    elif args.feedback_type == 'auto':
        prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'
    ft_rectify_prompt = open(prompt_filepath).read()

    ref_data = [json.loads(x) for x in open(args.refinement_file).readlines()]

    num_chatgpt_pass = 0
    if not args.no_output_gold_data:

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
            chatgpt_gold_code = ex["chatgpt_gold"]
            
            data_idx = data["task_id"].index(task_id)

            # Get the original reformatted MBPP prompt
            orig_prompt = format_prompt(data, task_id)

            # Remove method signature prefix
            gold_code = data[data_idx]["code"]
            # 

            pass_value, _ = eval(data, task_id=task_id, completion=chatgpt_gold_code)
            if not pass_value:
                print (task_id, "did not pass")
                chatgpt_gold_code = None
            else:
                num_chatgpt_pass += 1
 
            gold_code, _, _ = remove_header_from_code(gold_code, orig_prompt)
            if gold_code is None:
                continue

            if chatgpt_gold_code:
                chatgpt_gold_code,_, prompt_chatgpt = remove_header_from_code(chatgpt_gold_code, orig_prompt)
                if chatgpt_gold_code is None:
                    continue

            if gold_code is None:
                logging.warning(
                    f"Could not find function signature {func_sig} in gold code.\nGold code:\n{gold_code}"
                )
                continue
            nocg_ft_data["finetuning_prompt"].append(orig_prompt)
            nocg_ft_data["finetuning_completion"].append(gold_code)
            nocg_ft_data["task_id"].append(task_id)

            ft_data["finetuning_prompt"].append(orig_prompt)
            ft_data["finetuning_completion"].append(gold_code)
            ft_data["task_id"].append(task_id)
        
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

        ft_data.to_json(
            f"{args.output_dir}/finetuning_prompts_mbpp_gold_{args.output_file_suffix}.jsonl"
        )

        nocg_ft_data.to_json(
            f"{args.output_dir}/finetuning_prompts_mbpp_gold_{args.nocg_output_file_suffix}.jsonl"
        )


    rec_ft_data = {
        "finetuning_prompt": [],
        "finetuning_completion": [],
        "task_id": [],
    }

    num_passed = 0
    
    for ex in ref_data:
        task_id = ex["task_id"]
        prompt = format_prompt(data, task_id)
        orig_completion = ex["original_completion"]

        feedback, rectified = parse_chatgpt_output(ex["chatgpt_output"], orig_completion=orig_completion)

        orig_pass_value, orig_return_value = eval(data, task_id=task_id, completion=orig_completion)

        pass_value, return_value = eval(data, task_id=task_id, completion=rectified)
        if not pass_value:
            print (task_id, "did not pass")
            continue
        num_passed += 1

        error_text = get_error_text(return_value=orig_return_value)

        prompt_without_head = '"""'.join(prompt.split('"""')[:-1])+'"""'
        rectified, func_sig, _ = remove_header_from_code(rectified, prompt)
        if rectified is None:
            continue

        to_replace = {}
        to_replace["<<TASK>>"] = prompt_without_head 
        to_replace["<<CODE>>"] = orig_completion
        to_replace["<<HEADER>>"] = func_sig
        to_replace["<<FEEDBACK>>"] = feedback
        to_replace["<<ERROR>>"] = error_text

        rectify_prompt = fill_ft_prompt(ft_rectify_prompt, to_replace=to_replace)

        rec_ft_data["task_id"].append(task_id)
        rec_ft_data["finetuning_prompt"].append(rectify_prompt)
        rec_ft_data["finetuning_completion"].append(rectified)

        print ("RECTIFY PROMPT: ", rectify_prompt, '\n---')
        print (rectified, "\n========================================\n\n")

    print ("% passed : ", 100.0*num_passed/float(len(ref_data)))
    print ("% passed (chatgpt independent) ", 100.0*num_chatgpt_pass/float(len(ref_data)))
    rec_ft_data = Dataset.from_dict(rec_ft_data)

    rec_ft_data.to_json(
        f"{args.output_dir}/finetuning_prompts_mbpp_chatgpt_rectification_{args.output_file_suffix}.jsonl"
    )


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
    
    parser.add_argument("--output-file-suffix", type=str, default="")

    parser.add_argument("--nocg-output-file-suffix", type=str, default="nochatgpt")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args(None)
    create_prompts(args)


if __name__ == "__main__":
    main()
