import argparse
import gzip
import io
import itertools
import json
import pprint
import numpy as np
import re
import sys
import timeout_decorator
import traceback
from collections import Counter 
from io import StringIO
import sys
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from multiprocessing import Process, Queue
from tqdm import tqdm
from typing import Dict, List, Union
import os 
import ast 
import random 
from multiprocessing_utils import NoDaemonPool

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model completions on the MBPP benchmark."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File containing columns <args.prompt_column_name>, <args.prompt_column_name>, and 'task_id'.",
    )
    parser.add_argument("--k", default=None)
    parser.add_argument("--file-suffix", default="results")
    parser.add_argument(
        "--prompt-column-name", default="prompt", help="Name of prompt column."
    )
    parser.add_argument(
        "--completion-column-name",  default="completion", help="Column name containing completion."
    )
    args = parser.parse_args()
    return args


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Taken from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py#L13.
    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def compute_results(eval_results, shot):
    results = defaultdict(list)
    for row in eval_results:
        ti = row["task_id"]
        if shot not in row["passed"]:
            print ("Cannot find shot ", shot, " using passed of previous shot i.e ",row["passed"][str(int(shot)-1)])
            passed = row["passed"][str(int(shot)-1)]
        else:
            passed = row["passed"][shot]
        results[ti].append(passed)
    outputs = {
        ti: {"num_correct": np.sum(r), "num_total": len(r)} for ti, r in results.items()
    }
    return outputs


def compute_at_least_one_pass_per_task(results):
    total = 0
    task_ids = []
    for task_id, results_dict in results.items():
        if results_dict["num_correct"] > 0:
            total += 1
            task_ids.append(task_id)
    return total, task_ids


def compute_pass_at_ks(results, ks):
    output = {
        k: estimate_pass_at_k(
            [x["num_total"] for _, x in results.items()],
            [x["num_correct"] for _, x in results.items()],
            k,
        ).mean()
        for k in ks
    }
    return output


@timeout_decorator.timeout(3)
def eval_code(q, src, test, entry_point):
    standard_import = "from typing import List, Tuple, Any, Dict, Iterator, Callable"
    all_src = f"{standard_import}\n{src}\n{test}\ncheck({entry_point})\n"
    try:
        #print ("To execute:\n")
        #print (all_src)
        exec(all_src, {})
    except Exception:
        with io.StringIO() as f:
            traceback.print_exception(*sys.exc_info(), file=f)
            q.put((False, f.getvalue()))
        return
    q.put((True, None))


@timeout_decorator.timeout(5)
def eval_code_wrapper(src, test, entry_point):
    queue = Queue()
    p = Process(target=eval_code, args=(queue, src, test, entry_point))
    p.start()
    p.join(3)
    if p.is_alive():
        p.kill()
    if not queue.empty():
        return queue.get()
    else:
        return False, f"Exit code: {p.exitcode}"


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def check_in_prompt(output, input_args, prompt):
    if output in prompt:
        return True 
    elif input_args in prompt:
        return True 
    else:
        try:
            input_args = ast.literal_eval(input_args)
        except:
            input_args = [x.strip() for x in input_args.split(",")]
        if type(input_args)!=tuple:
            input_args = [input_args]
        prompt_lines = prompt.split("\n")
        for line in prompt_lines:
            if all([str(arg) in line for arg in input_args]):
                return True 
        return False 


def truncate_tests(tests, mode, num_tests):
    if mode=="eval":
        return tests 
    tests = tests.split('def check(candidate):')
    prefix = tests[0]+'\ndef check(candidate):\n'
    test_list = [l for l in tests[1].strip('\n').split('\n') if len(l.strip())>0]
    test_sublist = test_list[:num_tests]
    test_sublist = '\n'.join(test_sublist)
    tests = prefix+test_sublist+'\n'
    return tests 




def get_all_tests(data, task_id, mode=None, use_codeT=False, num_tests=None):
    #assert not use_codeT
    idx = data["task_id"].index(task_id)
    if mode=="eval":
        tests =  data['test'][idx] 
    elif mode=="infer_all":
        tests =  data['test_all'][idx] 
    elif mode=="infer_seen":
        tests = data['test_seen'][idx] 

    if use_codeT:
        codet_tests = data["test_input_args"][idx]
        num_tests = min(10, len(codet_tests))
        codet_tests = "\n".join(random.sample(codet_tests, num_tests))
        tests = codet_tests+"\n"+tests 
    if num_tests: 
        tests = truncate_tests(tests, mode, num_tests)
    return tests

def get_error_text(return_value):
    if return_value is None:
        return ''
    error_text = return_value['error_type']
    if return_value['error_type'] == "AssertionError":
        try:
            input = return_value['error']['input']
            output = return_value['error']['output']
            expected = return_value['error']['expected']
            error_text = return_value['error_type']+"\nINPUT: "+input+"\nOUTPUT: "+output+"\nEXPECTED: "+expected
        except:
            error_text = return_value['error_type']
    else:
        error_text = return_value['error_type']+" : "+return_value['error']
    return error_text

def get_entry_point(data, task_id):
    idx = data["task_id"].index(task_id)
    entrypoint = data["entry_point"][idx]
    return entrypoint



def get_dict_list(filename: str) -> List[Dict]:
    output_list = []
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        output_list.append(json.loads(line))
    elif filename.endswith(".jsonl"):
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    output_list.append(json.loads(line))
    elif filename.endswith(".csv"):
        d = load_dataset("csv", data_files={"train": filename})["train"]
        for i in range(len(d[d.column_names[0]])):
            output_list.append({col: d[col][i] for col in d.column_names})
    else:
        raise ValueError(f"Unrecognized file extension type for file {filename}!")
    return output_list


def eval(data, task_id, completion, mode="infer_seen", use_codeT=False, num_tests=None):
    #mode should be "eval" or "infer"
    if mode.lower() not in ["eval", "infer_all", "infer_seen"]:
       raise ValueError("Mode should be 'eval' or 'infer_all' or 'infer_seen'")
    entrypoint = get_entry_point(data, task_id)
    test_str = get_all_tests(data, task_id,mode=mode, use_codeT=use_codeT, num_tests=num_tests)
    try:
        p, r = eval_code_wrapper(completion, test_str, entrypoint)
    except Exception as e:
        with io.StringIO() as f:
            traceback.print_exception(*sys.exc_info(), file=f)
            r = f.getvalue()
        p = False
        print(f"Caught exception from eval_code: {e}\n{r}")
    return_val = None 
    #print ("RESULT:\n",r)
    if mode=="eval":
        r = None
    if r is not None:
        r = r.strip("\n").split("\n")[-1]
        error_type = r.split(":")[0]
        r = ":".join(r.split(":")[1:])
        return_val = {'error_type': error_type, 'error':r}
        if error_type == 'AssertionError':
            try:
                r = r.replace('Input:','::').replace('Output:','::').replace('Expected:','::').split('::')
                #if len(r)<4:
                #    print ("Error:\n", r)
                input = r[1].strip()
                output = r[2].strip()
                expected = r[3].strip()
                return_val['error'] = {'input': input, 'output': output, 'expected': expected}
            except:
                return_val['error'] = ''
    return p, return_val

def eval_pool(p_args):
    data, task_id, completion_i = p_args
    return eval(data, task_id, completion_i, mode="eval")

def eval_samples(args):
    
    output_file_prefix = args.input_file + f"_{args.file_suffix}"
    ext = args.input_file.split(".")[-1]
    output_file = f"{output_file_prefix}.{ext}"
    output_summ_file = f"{output_file_prefix}_summary.{ext}"

    if os.path.exists(output_file):
        samples = [json.loads(x) for x in open(output_file).readlines()]
        shots = set(samples[0]['passed'].keys())
    else:
        data = load_dataset("HumanEval")
        data = concatenate_datasets([data[k] for k in data.keys()])
        samples = get_dict_list(args.input_file)

        error_types = {}

        # ----
        eval_args = []
        for sample_dict in tqdm(samples, desc="vI, Evaluating and scoring..."):
            task_id = sample_dict["task_id"]
            completion = sample_dict[args.completion_column_name]
            prompt = sample_dict[args.prompt_column_name]
            p_dict = {}
            r_dict = {}
            for shot, completion_i in completion.items():
                eval_args.append([data, task_id, completion_i])
        
        with NoDaemonPool(os.cpu_count()) as p:
            eval_res = p.map(eval_pool, eval_args)

        eval_idx = 0
        for sample_dict in tqdm(samples, desc="vII, Evaluating and scoring..."):
            task_id = sample_dict["task_id"]
            completion = sample_dict[args.completion_column_name]
            p_dict = {}
            r_dict = {}
            for shot, completion_i in completion.items():
                p, r = eval_res[eval_idx]
                eval_idx += 1
                p_dict[shot] = p 
                r_dict[shot] = r 
                if shot not in error_types:
                    error_types[shot] = []
                if r:
                    error_types[shot].append(r['error_type'])
                else:
                    error_types[shot].append('None')
            sample_dict["passed"] = p_dict
            sample_dict["result"] = r_dict
        # ----
        
        
        # for sample_dict in tqdm(samples, desc="Evaluating and scoring..."):
        #     task_id = sample_dict["task_id"]
        #     completion = sample_dict[args.completion_column_name]
        #     prompt = sample_dict[args.prompt_column_name]
        #     p_dict = {}
        #     r_dict = {}
        #     for shot, completion_i in completion.items():
        #         #completion_i = get_clean_completion(prompt=prompt, completion=completion_i)
        #         p, r = eval(data, task_id, completion_i, mode="eval")
        #         p_dict[shot] = p 
        #         r_dict[shot] = r 
        #         if shot not in error_types:
        #             error_types[shot] = []
        #         if r:
        #             error_types[shot].append(r['error_type'])
        #         else:
        #             error_types[shot].append('None')
        #     sample_dict["passed"] = p_dict
        #     sample_dict["result"] = r_dict
        shots = set(error_types.keys())
    pass_at_k_results = {}
    if args.k is not None:
        ks = [int(elem) for elem in args.k.split(",")]
    else:
        ks = [min(Counter([sample['task_id'] for sample in samples]).values())]
        print ("Evaluating with k as ", ks)
    for shot in sorted(shots):
        print ("\nShot ", shot)
        # import pdb;pdb.set_trace()
        #error_types_i = Counter(error_types[shot])
        num_corr_results = compute_results(samples, shot)
        pass_at_k_results[shot] = compute_pass_at_ks(num_corr_results, ks)
        at_least_one_correct, _ = compute_at_least_one_pass_per_task(num_corr_results)
        pc_one_correct = at_least_one_correct / len(num_corr_results.keys())
        pass_at_k_results["% tasks with at least one passed completion"] = pc_one_correct
        print("\t", {key: round(val*100.0, 2) for key, val in pass_at_k_results[shot].items()})

        #print ("\t",error_types_i.most_common())

    with open(output_file, "w") as f:
        for d in samples:
            f.write(json.dumps(d) + "\n")
    with open(output_summ_file, "w") as f:
        f.write(json.dumps(pass_at_k_results))


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    eval_samples(args)


if __name__ == "__main__":
    main(parse_args())
