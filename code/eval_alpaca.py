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
from pytest import approx 
import ast 
import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from multiprocessing_utils import NoDaemonPool
from testing_utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model completions on the MBPP benchmark."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File containing columns <args.prompt_column_name>, <args.prompt_column_name>, and 'task_id'.",
    )
    parser.add_argument("--k", default="1,10")
    parser.add_argument("--file-suffix", default="results")
    parser.add_argument(
        "--prompt-column-name", default="prompt", help="Name of prompt column."
    )
    parser.add_argument(
        "--completion-column-name",  default="completion", help="Column name containing completion."
    )

    parser.add_argument(
        "--auxiliary-column-name", default=None, help="Any additional column required for evaluation."
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


def compute_results(eval_results):
    results = defaultdict(list)
    for row in eval_results:
        ti = row["task_id"]
        passed = bool(row["passed"])
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

def eval_code(q, src, test):
    import_stmts = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    exp_output = None 
    src = import_stmts + src
    input = test['input'].strip()
    input_args = input[input.find("("): input.rfind(")")+1]
    input = "output = "+input 
    if 'output' in test:
        exp_output = test['output']
    all_src = f"{src}\n{input}\n"
    try:
        buffer = StringIO()
        sys.stdout = buffer
        loc = {}
        exec(all_src, loc)
        buffer_value = buffer.getvalue()
        sys.stdout = sys.__stdout__ 
        out = loc['output']
        if out is None:
            out = buffer_value
        out = str(out)
        if exp_output is not None:
            error_msg = "Input: "+input_args+" Output: "+out+" Expected: "+exp_output
            if is_float(exp_output) and is_float(out):
                assert float(out) == approx(float(exp_output)), error_msg
            else:
                try:
                    out_lit = ast.literal_eval(out)
                    exp_out_lit = ast.literal_eval(exp_output)
                    assert out_lit == exp_out_lit, error_msg
                except:
                    assert out == exp_output, error_msg
    except Exception:
        with io.StringIO() as f:
            traceback.print_exception(*sys.exc_info(), file=f)
            q.put((False, f.getvalue()))
        return
    q.put((True, out))


@timeout_decorator.timeout(5)
def eval_code_wrapper(src, test):
    queue = Queue()
    p = Process(target=eval_code, args=(queue, src, test))
    p.start()
    p.join(3)
    if p.is_alive():
        p.kill()
    if not queue.empty():
        ret = queue.get()
        queue.close()
        return ret
    else:
        queue.close()
        return False, f"Exit code: {p.exitcode}"


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def trim(text, max_len):
    return " ".join(text.split(" ")[:max_len])

def get_error_text(return_value):
    if return_value is None:
        print ("Got return value None")
        return ''
    error_text = return_value['error_type']
    if return_value['error_type'] == "AssertionError":
        try:
            input = return_value['error']['input']
            output = return_value['error']['output']
            expected = return_value['error']['expected']
        except:
            return return_value['error_type']
        input = trim(input, 50)
        output = trim(output, 50)
        expected = trim(expected, 50)
        error_text = return_value['error_type']+"\nINPUT: "+input+"\nOUTPUT: "+output+"\nEXPECTED: "+expected
    else:
        error_text = return_value['error_type']+" : "+return_value['error']
    return error_text


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


def truncate_code(completion):
    if "REFINEMENT:" in completion or "Refinement:\n" in completion:
        refinement_str = (
            "REFINEMENT:" if "REFINEMENT:" in completion else "Refinement:\n"
        )
        ref_end_idx = completion.rfind(refinement_str) + len(refinement_str)
        completion = completion[ref_end_idx:]

    if "RECTIFIED CODE" in completion or "RECTIFIED CODE\n" in completion:
        refinement_str = (
            "RECTIFIED CODE" if "RECTIFIED CODE" in completion else "RECTIFIED CODE\n"
        )
        ref_end_idx = completion.rfind(refinement_str) + len(refinement_str)
        completion = completion[ref_end_idx:]

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

        completion = "\n".join(code)
    return completion 

def eval(completion, tests):
    completion = truncate_code(completion)
    passes = []
    returns = []
    for test in tests:
        #print (test)
        try:
            p, r = eval_code_wrapper(completion, test)
        except Exception as e:
            with io.StringIO() as f:
                traceback.print_exception(*sys.exc_info(), file=f)
                r = f.getvalue()
            p = False
            print(f"Caught exception from eval_code: {e}\n{r}")
            # if "No space left on device" in r:
            #     import pdb;pdb.set_trace()
        passes.append(p)
        returns.append(r)
    return_val = None 
    for p, r in zip(passes, returns):
        if not p and r is not None:
            r_org = r 
            error_type = r[:r.rfind(": Input:")].split("\n")[-1].split(" ")[-1]
            r = r[r.rfind(": Input:"):]
            if len(r)<5:
                error_type = r_org.strip("\n").split("\n")[-1]
                r = ''
            #print ("ERRORTYPE:", error_type)
            #print ("ERROR: ", r)
            return_val = {'error_type': error_type, 'error':r}
            if error_type == 'AssertionError':
                try:
                    r = r.replace('Input:','::').replace('Output:','::').replace('Expected:','::').split('::')
                    input = r[1].strip()
                    output = r[2].strip()
                    expected = r[3].strip()
                    return_val['error'] = {'input': input, 'output': output, 'expected': expected}
                except:
                    return_val['error'] = ''
            return p, return_val
    
    # print ("\n\n-----------------\n\nPrompt: ", prompt)
    # print ("\nCode:\n", completion)
    # print ("passed: ", p, "\nresult: ", r)
    return "True", None 

def eval_pool(p_args):
    completion, tests = p_args
    return eval(completion, tests)

def eval_samples(args):
    ks = [int(elem) for elem in args.k.split(",")]
    output_file_prefix = args.input_file + f"_{args.file_suffix}"
    ext = args.input_file.split(".")[-1]
    output_file = f"{output_file_prefix}.{ext}"
    output_summ_file = f"{output_file_prefix}_summary.{ext}"

    data = load_dataset("ALP/code-alpaca-mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])
    data_task_to_index_dict = {data['task_id'][i]: i for i in range(len(data['task_id']))}
    samples = get_dict_list(args.input_file)

    print ("Number of instances originally: ", len(samples))
    
    error_types = []

    num_passed = 0
    count = 0 

    if os.path.exists(output_file):
        samples = [json.loads(x) for x in open(output_file).readlines()]
    else:
        eval_args = []
        for i, sample_dict in enumerate(samples):
            task_id = sample_dict["task_id"]
            index = data_task_to_index_dict[task_id]
            tests = data[index]['test_list']
            completion = sample_dict[args.completion_column_name]
            prompt = sample_dict[args.prompt_column_name]

            #print ("\n================================\n\nPROMPT:", prompt )
            #print ("COMPLETION:\n"+completion, '\n')
            eval_args.append((completion, tests))
        
        eval_res = []
        with NoDaemonPool(os.cpu_count()//2) as p:
            eval_res = p.map(eval_pool, eval_args)
        
        for (p,r), sample_dict in zip(eval_res, samples):
            # p, r = eval(completion, tests)
            if p:
                num_passed += 1 
                # print (num_passed, 'at ', count)
            # else:
            #     print (r)
            # if count %100==0:
            #     print (num_passed, 'at ', count)
            sample_dict["passed"] = p
            sample_dict["result"] = r
            if r:
                error_types.append(r['error_type'])
            else:
                error_types.append('None')
            # count += 1

        with open(output_file, "w") as f:
            for d in samples:
                f.write(json.dumps(d) + "\n")

    error_types = Counter(error_types)
    num_corr_results = compute_results(samples)
    pass_at_k_results = compute_pass_at_ks(num_corr_results, ks)
    at_least_one_correct, _ = compute_at_least_one_pass_per_task(num_corr_results)
    pc_one_correct = at_least_one_correct / len(num_corr_results.keys())
    pass_at_k_results["% tasks with at least one passed completion"] = pc_one_correct
    print(pass_at_k_results)

    print (error_types.most_common())
    with open(output_summ_file, "w") as f:
        f.write(json.dumps(pass_at_k_results))


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    eval_samples(args)


if __name__ == "__main__":
    main(parse_args())
