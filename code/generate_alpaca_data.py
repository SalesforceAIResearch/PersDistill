
import argparse
import json
import openai
import re 
import os
import pprint
import time
from datasets import load_dataset, concatenate_datasets
from eval_mbpp import eval, get_error_text
import sys 
import io 
import timeout_decorator
import traceback
from datasets import concatenate_datasets, load_dataset
from multiprocessing import Process, Queue
import numpy as np         
from io import StringIO
import sys
from eval_alpaca import eval_code_wrapper
from datasets import Dataset

def fill_ft_prompt(text, to_replace):
    for k, v in to_replace.items():
        text = text.replace(k, v)
    return text 

def initialize_openai(args):
    api_key = open(f"{args.openai_creds_dir}/openai_api_key.txt").read()
    openai.organization = open(
        f"{args.openai_creds_dir}/openai_organization_id.txt"
    ).read()
    openai.api_key = api_key


def sample_code_from_openai_model(args, prompt):
    output_strs = []
    start = time.time()

    arch_mapping = {
        "codex": "code-davinci-002",
        "gpt3": "text-davinci-001",
        "davinci-002": "text-davinci-002",
        "davinci-003": "text-davinci-003",
        "ada": "text-ada-001",
        "babbage": "text-babbage-001",
        "curie": "text-curie-001",
        "chatgpt": "gpt-3.5-turbo"
    }
    engine_name = arch_mapping[args.arch]

    for i in range(args.num_samples):
        while time.time() - start < args.max_request_time:
            try:
                if args.arch == "chatgpt":
                    response = openai.ChatCompletion.create(
                        model=engine_name,
                        messages=prompt
                    )
                    output_strs += [
                        choice["message"] for choice in response["choices"]
                    ]
                else:
                    response = openai.Completion.create(
                        engine=engine_name,
                        prompt=prompt,
                        max_tokens=2048,
                        n=1,
                        temperature=args.temperature,
                    )
                    output_strs += [
                        choice["text"] for choice in response["choices"]
                    ]
                break
            except Exception as e:
                print(
                    f"Unexpected exception in generating solution. Sleeping again: {e}"
                )
                time.sleep(args.sleep_time)
    
    return output_strs


def write_jsonl(data, output_filepath):
    with open(output_filepath, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    

def generate_unit_test_output_for_problems(args, code_filepath):

    data = [ json.loads(x) for x in open(code_filepath).readlines()]

    data_unittests = [] 

    for data_i in data:
        code = data_i['code']

        print ("\nTASK:: ", data_i['instruction'])
        print ("\nCODE::\n"+data_i['code'])
        print ("\nINPUTS::\n"+data_i['inputs'])

        entrypoint = None
        try:
            entrypoint = code.split("def ")[1].split('(')[0].strip()
        except Exception:
            print (code)
            traceback.print_exception(*sys.exc_info())
        unittests = []
        for input in data_i['inputs'].strip().split("\n"):
            input = input.strip()
            if not input.startswith(entrypoint):
                if entrypoint in input:
                    input = input[input.find(entrypoint):]
            test = {"input": input}
            #exec_code = "output = "+input.strip()
            try:
                pass_value, return_value = eval_code_wrapper(code, test)
            except Exception:
                traceback.print_exception(*sys.exc_info())
            try:
                if entrypoint is not None:
                    obj_regex = '<'+entrypoint+' object at .*>'
                else:
                    obj_regex = ' object at .*>'
                if re.search(obj_regex, return_value):
                    return_value = None 
            except Exception:
                traceback.print_exception(*sys.exc_info())

            unittests.append({'pass': pass_value, 'output': return_value, 'input': input})
        data_i['unittests'] = unittests

        
        print ("\nUNITTESTS::\n")
        for o in data_i['unittests']:
            for k,v in o.items():
                print (k, ':  ', v)
        print ("\n=================================\n\n")                

        data_unittests.append(data_i)
    
    return data_unittests


def generate_unit_test_input_for_problems(args, code_filepath):

    initialize_openai(args)

    args.arch = "chatgpt"

    prompt_filepath = 'prompts/chatgpt_unittest_instruction.txt'

    prompt_template = open(prompt_filepath).read()

    data = json.load(open(code_filepath))

    data_unittests = []

    if not args.start:
        args.start = 0 
    if not args.end:
        args.end = len(data)
    ids_in_range = range(args.start, args.end)
    if args.proc_id is not None:
        ids_in_range = np.array_split(ids_in_range, args.num_procs)[args.proc_id]
    print ("number of instances: ", len(ids_in_range))
    for i, data_i in enumerate(data): 
        if i not in ids_in_range:
            continue
        task = data_i['instruction']
        code = data_i['output']
        if "def " not in code:
            continue 
        to_replace = {}
        to_replace['<<TASK>>'] = task 
        to_replace['<<CODE>>'] = code 
        prompt = fill_ft_prompt(prompt_template, to_replace)
        prompt = [{"role": "user", "content": prompt}]
        
        inputs = sample_code_from_openai_model(args, prompt)[0]['content'].strip()

        print ("\nTASK:: ", task)
        print ("\nCODE::\n"+code)
        print ("\nINPUTS::\n"+inputs)

        d = {}
        d['task_id'] = i
        d['instruction'] = task 
        d['original_input'] = data_i['input']
        d['code'] = code 
        d['inputs'] = inputs
        print ("\n=================================\n\n")

        data_unittests.append(d)
    
    return data_unittests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch",
        default="gptj",
        choices=[
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

    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "-s", "--start", default=None, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=None, type=int, help="Task ID to end with (exclusive)."
    )

    parser.add_argument(
        "-p", "--num-procs", default=None, type=int, help="Number of processes to run (=number of gpus)"
    )

    parser.add_argument(
        "-i", "--proc-id", default=None, type=int, help="Process id (in [0, num_proc-1])"
    )

    parser.add_argument(
        "-n",
        "--num-shots",
        default=0,
        type=int,
        help="Number of assert (test examples) to give in the task description.",
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
        "--generate-unittest-output",
        default=False, action="store_true",
        help="Whether to also generate unittest outputs or not"
    )


    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    if args.proc_id is not None:
        args.output_file_suffix = args.output_file_suffix+"_"+str(args.proc_id)
    if args.start is not None and args.end is not None:
        start_end = f"{args.start}-{args.end}"   
    else:
        start_end = "" 
    output_filepath_stage1 = os.path.join(
        args.output_dir,
        f"unittests_input_{args.arch}_{args.num_shots}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl",
    )

    code_filepath = os.path.join(args.input_dir, args.input_file)
    if not os.path.exists(output_filepath_stage1):
        data_unittests = generate_unit_test_input_for_problems(args, code_filepath)
        write_jsonl(data_unittests, output_filepath_stage1)
    else:
        print ("Not re-running, Already found code output existing in: ", output_filepath_stage1)


    if args.generate_unittest_output:    
        output_filepath_stage2 = os.path.join(
            args.output_dir,
            f"unittests_{args.arch}_{args.num_shots}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl",
        )

        code_filepath = os.path.join(args.input_dir, args.input_file)
        if not os.path.exists(output_filepath_stage2):
            data_unittests = generate_unit_test_output_for_problems(args, output_filepath_stage1)
            write_jsonl(data_unittests, output_filepath_stage2)
        else:
            print ("Not re-running, Already found code output existing in: ", output_filepath_stage2)


if __name__ == "__main__":
    main(parse_args())
