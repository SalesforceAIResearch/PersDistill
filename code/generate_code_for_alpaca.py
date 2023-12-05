import argparse
import json
import logging
import os
import pprint
import re
import numpy as np 
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from eval_alpaca import eval, get_error_text
from file_utils import write_jsonl
from openai_utils import initialize_openai, sample_code_from_openai_model
import re 
import torch 
import deepspeed 
from transformers.models.codegen.modeling_codegen import CodeGenBlock
from hf_credentials import auth_token

def format_prompt(text, tests, sample_code):
    # Add code prefix
    fn_name = tests[0]['input'].split("(")[0]
    fn_search = re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", sample_code)
    if fn_search is None:
        return None 
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'"""\n{text}\n"""\n\n{code_prefix}\n'
    return prompt

def get_clean_completion(completion, prompt):
    if completion.startswith(prompt):
        completion = completion[len(prompt):]
    prompt_head = prompt.split('"""')[-1].split("def ")[0]
    func_sig = "def "+prompt.split('"""')[-1].split("def ")[1]
    if func_sig.strip() not in completion:
        completion = func_sig+"\n"+completion
    prompt_last_line = prompt.strip('\n').split('\n')[-1]
    fn_name = prompt_last_line[re.search(f"def ", prompt_last_line).start():].split(' ')[1].split('(')[0]
    try:
        completion = completion[re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", completion).start():]
        eof_m = re.search(r'\n[A-Za-z#"]+?', completion)
        if eof_m is not None:
            completion = completion[: eof_m.start() + 1]
    except:
        return completion
    completion = prompt_head+"\n"+completion.strip()
    return completion 


def fill_prompt_template(text, to_replace):
    for k, v in to_replace.items():
        text = text.replace(k, v)
    return text 


def sample_code_from_codegen(args, prompt, model, tokenizer, tests=None, num_samples=None, shot=1):
    completions = []
    if not num_samples:
        num_samples = args.num_samples 
    input_ids = tokenizer(
        prompt, truncation=True, max_length=100, return_tensors="pt"
    ).input_ids.cuda()
    if args.temperature == 0.0:
        args.num_samples = 1
    try:
        # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
        if args.temperature > 0:
            tokens = model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=args.num_samples,
                max_length=input_ids.shape[1] + 1024,
                temperature=args.temperature,
                top_p=0.95,
                use_cache=True,
            )
        else:
            tokens = model.generate(
                input_ids,
                num_return_sequences=1,
                max_length=input_ids.shape[1] + 1024,
                use_cache=True,
            )
    except RuntimeError as e:
        logging.error(f"Could not sample from model: {e}")
    completions = tokenizer.batch_decode(tokens)
    completions = [text[: text.find("<|endoftext|>")] if "<|endoftext|>" in text else text for text in completions]

    #for completion in completions:
    #    print ("original: ",completion)
    
    if shot == 1 and args.refine:

        #print ("\n\n==============================\nPROMPT:")
        #print (prompt)
        
        if args.feedback_type == 'nl':
            prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
        elif args.feedback_type == 'auto':
            prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'

        ft_refinement_prompt = open(prompt_filepath).read()
        refinements = []
        for completion in completions:
            #print ("ORIG CODE:")
            #print (completion)

            

            completion_to_eval = get_clean_completion(prompt=prompt, completion=completion)

            pass_value, return_value = eval(prompt=prompt, completion=completion_to_eval, tests=tests)
            if pass_value:
                refinements.append(completion)
                continue
            to_replace = {}
            to_replace["<<TASK>>"] = "" 
            to_replace["<<CODE>>"] = completion_to_eval
            header = prompt.split('"""')[-1]+"\n"
            to_replace["<<HEADER>>"] = header 

            if args.feedback_type == "auto":
                error_text = get_error_text(return_value=return_value)
                to_replace["<<ERROR>>"] = error_text

            refinement_prompt = fill_prompt_template(ft_refinement_prompt, to_replace=to_replace)

            #print("REFINEMNT PROMPT:")
            #print (refinement_prompt)
            refinement = sample_code_from_codegen(args, refinement_prompt, model, tokenizer, num_samples=1, shot=2)[0]
            refinement = refinement[len(refinement_prompt):]
            refinement = header + "\n" + refinement
            if  "code is correct" in refinement.lower() or refinement.lower().strip() == "none" or len(refinement.split("\n"))<3:
                refinements.append(completion)
            else:
                refinements.append(refinement)
        completions = refinements
        #print ("\nREFINEMENT:")
        #print (completions[0])
        #print ("\n-----------------------------------------------------------\n\n")

    #elif shot==1:
    #    print ("\nCOMPLETION:")
    #    print (completions[0])
    return completions

def resume_generate_feedback_for_problems(args, code_filepath, feedback_filepath):
    # outputs = []
    initialize_openai(args)

    outputs = [json.loads(line) for line in open(feedback_filepath).readlines()]
    seen_task_ids = set([d['task_id'] for d in outputs])

    args.arch = "chatgpt"

    data = load_dataset("ALP/code-alpaca-mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])
    data_task_to_index_dict = {data['task_id'][i]: i for i in range(len(data['task_id']))}

    instruction_turns = []

    if args.feedback_type == 'nl':
        prompt_filepath = 'prompts/chatgpt_feedback_instruction.txt'
    elif args.feedback_type == 'auto':
        if args.chatgpt_feedback_type == 'v1':
            prompt_filepath = 'prompts/chatgpt_error_instruction.txt'
        elif args.chatgpt_feedback_type == 'v2':
            prompt_filepath = 'prompts/chatgpt_error_instruction_v2.txt'
        elif args.chatgpt_feedback_type == 'v0':
            prompt_filepath = 'prompts/chatgpt_error_instruction_v0.txt'
    turn = []
    for line in open(prompt_filepath).readlines():
        if line.startswith("=") and len(set(line.strip()))==1:
            instruction_turns.append("\n".join(turn))
            turn = []
        else:
            turn.append(line.strip())

    code_data = [json.loads(line) for line in open(code_filepath).readlines()]
    for i, data_i in enumerate(code_data):
        prompt = data_i['prompt']
        completion = data_i['completion']
        task_id = data_i['task_id']
        index = data_task_to_index_dict[task_id]
        tests = data['test_list'][index]
        if task_id in seen_task_ids:
            continue

        pass_value, return_value = eval(completion=completion, tests=tests)
        # print(i)
        # continue
        if pass_value:
            continue 
        print(f"New instance {i}")
        error_text = None 
        prompt_header = prompt.split('"""')[-1]
        prompt_task = prompt[:-len(prompt_header)]

        to_replace = {"<<TASK>>": prompt_task, "<<HEADER>>": prompt_header, "<<CODE>>": completion}
        if len(instruction_turns)>1:
            turn1_prompt = instruction_turns[0]
            turn1_prompt = fill_prompt_template(turn1_prompt, to_replace)
            turn1_prompt = [{"role": "user", "content": turn1_prompt}]
            if args.get_openai_gold:
                #print ("\nCHATGPT PROMPT FOR GOLD:")
                #print (turn1_prompt[0]['content'])
                gold = sample_code_from_openai_model(args, turn1_prompt)[0]['content']
            else:
                gold = data_i['gold']
            turn1_prompt.append({"role": "assistant", "content": gold})
            turn2_prompt = instruction_turns[1]
        else:
            turn1_prompt = None 
            turn2_prompt = instruction_turns[0]
            gold = None 
            
        if not pass_value:
            error_text = get_error_text(return_value)
            to_replace["<<ERROR>>"] = error_text

        turn2_prompt = fill_prompt_template(turn2_prompt, to_replace)

        if turn1_prompt is None:
            turn2_prompt = [{"role": "user", "content": turn2_prompt}]
        else:
            turn2_prompt = turn1_prompt+[{"role": "user", "content": turn2_prompt}]

        try:
            feedback = sample_code_from_openai_model(args, turn2_prompt)[0]['content']
        except:
            print(f"feedback failed {i}")
            continue
        d = {
                "task_id": data_i['task_id'],
                "prompt": prompt,
                "original_completion": completion,
                "chatgpt_output": feedback
        }
        if args.get_openai_gold and gold:
            d["chatgpt_gold"] = gold
        else:
            d["chatgpt_gold"] = None

        if error_text:
            d["original_error_text"] = error_text

        outputs.append(d)
        if i%100==0:
            print ("Finished ",i)
        # print ("ORIGINAL COMPLETION:")
        # print (completion)
        # print ("\nCHATGPT PROMPT:")
        # print ("\n".join([t['content'] for t in turn2_prompt]))
        # print ("\nCHATGPT RECTIFICATION")
        # print (feedback)
        # if gold:
        #     print ("\nCHATGPT GOLD:")
        #     print (d["chatgpt_gold"])
        # print ("\n------------------------------------------\n\n")
    return outputs

def generate_feedback_for_problems(args, code_filepath):
    outputs = []
    initialize_openai(args)

    args.arch = "chatgpt"

    data = load_dataset("ALP/code-alpaca-mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])
    data_task_to_index_dict = {data['task_id'][i]: i for i in range(len(data['task_id']))}

    instruction_turns = []

    if args.feedback_type == 'nl':
        prompt_filepath = 'prompts/chatgpt_feedback_instruction.txt'
    elif args.feedback_type == 'auto':
        if args.chatgpt_feedback_type == 'v1':
            prompt_filepath = 'prompts/chatgpt_error_instruction.txt'
        elif args.chatgpt_feedback_type == 'v2':
            prompt_filepath = 'prompts/chatgpt_error_instruction_v2.txt'
        elif args.chatgpt_feedback_type == 'v0':
            prompt_filepath = 'prompts/chatgpt_error_instruction_v0.txt'

    turn = []
    for line in open(prompt_filepath).readlines():
        if line.startswith("=") and len(set(line.strip()))==1:
            instruction_turns.append("\n".join(turn))
            turn = []
        else:
            turn.append(line.strip())

    code_data = [json.loads(line) for line in open(code_filepath).readlines()]
    for i, data_i in enumerate(code_data):
        prompt = data_i['prompt']
        completion = data_i['completion']
        task_id = data_i['task_id']
        index = data_task_to_index_dict[task_id]
        tests = data['test_list'][index]
        pass_value, return_value = eval(completion=completion, tests=tests)
        # print(i)
        # continue
        if pass_value:
            continue 
        error_text = None 
        prompt_header = prompt.split('"""')[-1]
        prompt_task = prompt[:-len(prompt_header)]

        to_replace = {"<<TASK>>": prompt_task, "<<HEADER>>": prompt_header, "<<CODE>>": completion}
        for count in range(args.num_feedbacks):
            if len(instruction_turns)>1:
                turn1_prompt = instruction_turns[0]
                turn1_prompt = fill_prompt_template(turn1_prompt, to_replace)
                turn1_prompt = [{"role": "user", "content": turn1_prompt}]
                if args.get_openai_gold:
                    #print ("\nCHATGPT PROMPT FOR GOLD:")
                    #print (turn1_prompt[0]['content'])
                    gold = sample_code_from_openai_model(args, turn1_prompt)[0]['content']
                else:
                    gold = data_i['gold']
                turn1_prompt.append({"role": "assistant", "content": gold})
                turn2_prompt = instruction_turns[1]
            else:
                turn1_prompt = None 
                turn2_prompt = instruction_turns[0]
                gold = None 
                
            if not pass_value:
                error_text = get_error_text(return_value)
                to_replace["<<ERROR>>"] = error_text

            turn2_prompt = fill_prompt_template(turn2_prompt, to_replace)

            if turn1_prompt is None:
                turn2_prompt = [{"role": "user", "content": turn2_prompt}]
            else:
                turn2_prompt = turn1_prompt+[{"role": "user", "content": turn2_prompt}]

            try:
                feedback = sample_code_from_openai_model(args, turn2_prompt)[0]['content']
            except:
                continue
            d = {
                    "task_id": data_i['task_id'],
                    "prompt": prompt,
                    "original_completion": completion,
                    "chatgpt_output": feedback
            }
            if args.get_openai_gold and gold:
                d["chatgpt_gold"] = gold
            else:
                d["chatgpt_gold"] = None

            if error_text:
                d["original_error_text"] = error_text

            outputs.append(d)
        if i%100==0:
            print ("Finished ",i)
        # print ("ORIGINAL COMPLETION:")
        # print (completion)
        # print ("\nCHATGPT PROMPT:")
        # print ("\n".join([t['content'] for t in turn2_prompt]))
        # print ("\nCHATGPT RECTIFICATION")
        # print (feedback)
        # if gold:
        #     print ("\nCHATGPT GOLD:")
        #     print (d["chatgpt_gold"])
        # print ("\n------------------------------------------\n\n")
    return outputs





def generate_code_for_problems(args):
    data = load_dataset("ALP/code-alpaca-mbpp")
    print ("Dataset Loaded")
    data = concatenate_datasets([data[k] for k in data.keys()])
    output = []
    if args.arch in ["gpt3", "codex", "chatgpt"]:
        initialize_openai(args)
        prompt_format = open("prompts/chatgpt_instruction.txt").read()

    elif args.arch.startswith("codegen"): 
        tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.arch}-mono")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if args.arch == "codegen-6B":

            if args.model_path is None:
                model = AutoModelForCausalLM.from_pretrained(
                    f"Salesforce/{args.arch}-mono",
                    device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, device_map="auto")
                
        elif args.arch == "codegen-16B":

            if args.model_path is None:
                model = AutoModelForCausalLM.from_pretrained(
                    f"Salesforce/{args.arch}-mono", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", pad_token_id=tokenizer.eos_token_id)
                #checkpoint_json = None
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", pad_token_id=tokenizer.eos_token_id)
                
            print ("Model Loaded")

    elif args.arch.startswith("starcoder"):
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, use_auth_token=auth_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=auth_token)

    task_ids_range = [data['task_id'][i] for i in range(len(data))] 
    if args.start is not None and args.end is not None:
        task_ids_range = list(range(args.start, args.end))
    if args.exclude_start is not None and args.exclude_end is not None:
        task_ids_range = list(range(args.start, args.exclude_start))+list(range(args.exclude_end, args.end))
    if args.proc_id is None: 
        indices_filtered = [i for i in range(len(data)) if data["task_id"][i] in task_ids_range]
        data_filtered = data.select(indices_filtered)
    elif args.proc_id is not None:
        task_ids_range = np.array_split(task_ids_range, args.num_procs)[args.proc_id]
        indices_filtered = [i for i in range(len(data)) if data["task_id"][i] in task_ids_range]
        data_filtered = data.select(indices_filtered)


    prompts_batch = []
    taskids_batch = []
    gold_batch = []
    for i in tqdm(range(len(data_filtered))):
        prompt = format_prompt(
            data_filtered["text"][i],
            data_filtered["test_list"][i],
            data_filtered["code"][i]
        )
        gold = data_filtered["code"][i]
        if prompt is None:
            continue
        task_id = data_filtered["task_id"][i]
        prompts_batch.append(prompt)
        taskids_batch.append(task_id)
        gold_batch.append(gold)
        if len(prompts_batch) == args.batch_size:
            if args.arch.startswith("codegen") or args.arch.startswith("starcoder"):
                completions = sample_code_from_codegen(args, prompt=prompts_batch, model=model, tokenizer=tokenizer)   
            
            for i in range(len(completions)):
                prompt = prompts_batch[i]
                task_id = taskids_batch[i]
                completion = completions[i]
                gold = gold_batch[i]
                clean_completion = get_clean_completion(completion, prompt)
                #print ("clean completion: ",clean_completion, "\n\n")
                if len(clean_completion.strip())>0:
                   completion = clean_completion
                if completion is None:
                   continue 
                output.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt,
                        "gold": gold,
                        "completion": completion,
                    }
                )
            taskids_batch = []
            prompts_batch = []
            gold_batch = []
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch",
        default="gptj",
        choices=[
            "starcoder",
            "gptj",
            "codex",
            "gpt3",
            "chatgpt",
            "codegen-16B",
            "codegen-6B",
            "davinci-002",
            "davinci-003",
            "ada",
            "babbage",
            "curie",
        ],
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Directory to load model checkpoint from. If None, will load a pre-trained "
        "CodeGen model using the --arch argument instead.",
    )

    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Which data split to use. By default 'test'.",
    )
    parser.add_argument(
        "-s", "--start", default=None, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=None, type=int, help="Task ID to end with (exclusive)."
    )

    parser.add_argument(
        "-exs", "--exclude-start", default=None, type=int, help="TASK ID to start excluding from."
    )

    parser.add_argument(
        "-exe", "--exclude-end", default=None, type=int, help="TASK ID to end excluding."
    )

    parser.add_argument(
        "-p", "--num-procs", default=None, type=int, help="Number of processes to run (=number of gpus)"
    )

    parser.add_argument(
        "-i", "--proc-id", default=None, type=int, help="Process id (in [0, num_proc-1])"
    )

    parser.add_argument(
        "--local_rank",
        default=None,
        type=int,
        help="Rank of process",
    )

    parser.add_argument(
        "--num-demos",
        default=None,
        type=int,
        help="Number of examples from doc string to show in prompt.",
    )

    parser.add_argument(
        "--feedback-type",
        type=str,
        help="type of feedback needed: nl or auto"
    )

    parser.add_argument(
        "--chatgpt-feedback-type",
        type=str,
        default='v1',
        help="type of chatgpt feedback prompt: v1 or v2 or v0"
    )

    parser.add_argument(
        "--num-feedbacks",
        type=int,
        default=1,
        help="Number of rectified code to generate from chatgpt.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for generation.",
    )
    parser.add_argument(
        "--max-request-time",
        type=int,
        default=80,
        help="Max. time to wait for a successful GPT-3 request.",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="Time to sleep (in seconds) between each GPT-3 call.",
    )
    parser.add_argument(
        "--openai-creds-dir",
        type=str,
        default='openai_creds',
        help="Directory where OpenAI API credentials are stored. Assumes the presence of "
        "openai_api_key.txt and openai_organization_id.txt files.",
    )

    parser.add_argument(
        "--get-openai-feedback",
        action="store_true",
        help="Whether to get feedback from openai GPT models on the generated code"
    )

    parser.add_argument(
        "--get-openai-gold",
        action="store_true",
        help="whether to get gold data from openai GPT models or not"
    )

    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether to further refine the code generation or not"
    )

    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.batch_size > 1:
        print("batch-size > 1 is not working properly and results in some empty outputs. Using batch size =1 instead")
        args.batch_size = 1

    if args.start is not None and args.end is not None:
        start_end = f"{args.start}-{args.end}"
    else:
        start_end = ""
    if args.proc_id is not None:
        args.output_file_suffix = args.output_file_suffix+"_"+str(args.proc_id)

    output_filepath = os.path.join(
        args.output_dir,
        f"samples_{args.split}_{args.arch}_{args.num_samples}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl",
    )
    if not os.path.exists(output_filepath):
        completions = generate_code_for_problems(args)
        write_jsonl(completions, output_filepath)
    else:
        print ("Not re-running, Already found code output existing in: ", output_filepath)

    if args.get_openai_feedback:
        feedback_filepath = os.path.join(
            args.output_dir,
            f"chatgpt_feedback_{args.split}_{args.arch}_{args.num_samples}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl"
        )
        if not os.path.exists(feedback_filepath):
            feedbacks = generate_feedback_for_problems(args, output_filepath)
            write_jsonl(feedbacks, feedback_filepath)
        else:
            print ("Resume re-runnning, start fomr already found openai feedback existing in: ", feedback_filepath)
            feedbacks = resume_generate_feedback_for_problems(args, output_filepath, feedback_filepath)
            feedback_filepath = os.path.join(
                args.output_dir,
                f"chatgpt_new_feedback_{args.split}_{args.arch}_{args.num_samples}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl"
            )
            write_jsonl(feedbacks, feedback_filepath)

if __name__ == "__main__":
    main(parse_args())
