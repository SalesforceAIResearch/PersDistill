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
    parser.add_argument("--k", default="1,10")
    parser.add_argument("--file-suffix", default="results")
    parser.add_argument(
        "--prompt-column-name", default="prompt", help="Name of prompt column."
    )
    parser.add_argument(
        "--completion-column-name",  default="completion", help="Column name containing completion."
    )

    parser.add_argument(
        "-s", "--start", default=1, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=975, type=int, help="Task ID to end with (exclusive)."
    )

    parser.add_argument(
        "-exs", "--exclude-start", default=None, type=int, help="TASK ID to start excluding from."
    )

    parser.add_argument(
        "-exe", "--exclude-end", default=None, type=int, help="TASK ID to end excluding."
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
        if n < k:
            return 0.0
        elif n - c < k:
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
    all_src = f"{src}\n{test}\ncheck({entry_point})\n"
    try:
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


# def is_float(element: str) -> bool:
#     try:
#         float(element)
#         return True
#     except ValueError:
#         return False


# def format_test(data, entrypoint, task_id):
#     idx = data["task_id"].index(task_id)
#     test_list = data["test_list"][idx]

#     test_str = "def check(candidate):\n"

#     # use pytest.approx() for float results
#     if is_float(test_list[0].split("==")[-1]):
#         answer_float = True
#     else:
#         answer_float = False 
#     test_str = "from pytest import approx\n\n" + test_str
#     for i in range(len(test_list)):
#         split = test_list[i].split("==")
#         func_sig =  " ".join("==".join(split[:-1]).split(' ')[1:])
#         input_args = func_sig.split(entrypoint)[-1].replace("'", '"').strip()
#         if answer_float:
#             assert_stmt = "assert x == approx(y)"
#         else:
#             assert_stmt = "assert x == y"
#         split[-1] = f"; y = {split[-1].strip()}; {assert_stmt}, 'Input: {input_args} Output: '+str(x)+' Expected: '+str(y)"
#         test_list[i] =  "x = "+func_sig+ split[-1]

#     for test in test_list:
#         test_str += f"\t{test}\n"
#     test_str += "\n"

#     if entrypoint != "check":
#         test_str = test_str.replace(entrypoint, "candidate")
#     else:
#         test_str = test_str.replace(f"assert {entrypoint}", "candidate")
#     return test_str

def get_all_tests(data, task_id, num_tests=None):
    idx = data["task_id"].index(task_id)
    tests = data["test_formatted"][idx]
    if num_tests is not None:
        tests = tests[:num_tests]
    return tests 

def get_error_text(return_value):
    if return_value is None:
        return ''
    error_text = return_value['error_type']
    if return_value['error_type'] == "AssertionError":
        input = return_value['error']['input']
        output = return_value['error']['output']
        expected = return_value['error']['expected']
        error_text = return_value['error_type']+"\nINPUT: "+input+"\nOUTPUT: "+output+"\nEXPECTED: "+expected
    else:
        error_text = return_value['error_type']+" : "+return_value['error']
    return error_text

def get_entry_point(data, task_id):
    idx = data["task_id"].index(task_id)
    return data['entrypoint'][idx]



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


def eval(data, task_id, completion, num_tests=None):
    completion = truncate_code(completion)
    entrypoint = get_entry_point(data, task_id)
    test_str = get_all_tests(data, task_id, num_tests)
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
    if r is not None:
        r = r.strip("\n").split("\n")[-1]
        error_type = r.split(":")[0]
        r = ":".join(r.split(":")[1:])
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
    # print ("\n\n-----------------\n\nPrompt: ", prompt)
    # print ("\nCode:\n", completion)
    # print ("passed: ", p, "\nresult: ", r)
    return p, return_val

def eval_pool(p_args):
    data, task_id, completion = p_args
    return eval(data, task_id, completion)

def eval_samples(args):
    ks = [int(elem) for elem in args.k.split(",")]
    output_file_prefix = args.input_file + f"_{args.file_suffix}"
    ext = args.input_file.split(".")[-1]
    output_file = f"{output_file_prefix}.{ext}"
    output_summ_file = f"{output_file_prefix}_summary.{ext}"

    mbpp_orig = load_dataset('mbpp', 'sanitized')
    mbpp_orig = concatenate_datasets([mbpp_orig['test'], mbpp_orig['validation'], mbpp_orig['prompt']])
                                     
    sanitized_task_ids = set(d['task_id'])
    assert sanitized_task_ids  == set([2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 77, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 113, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 137, 138, 139, 140, 141, 142, 143, 145, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 222, 223, 224, 226, 227, 228, 229, 230, 232, 233, 234, 235, 237, 238, 239, 240, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 290, 291, 292, 293, 294, 295, 296, 297, 299, 300, 301, 304, 305, 306, 307, 308, 309, 310, 311, 312, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 468, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 554, 555, 556, 557, 558, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 572, 573, 574, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600])

    if os.path.exists(output_file):
        samples = [json.loads(x) for x in open(output_file).readlines()]
        samples = [x for x in samples if x['task_id'] in sanitized_task_ids]
        shots = set(samples[0]['passed'].keys())
    else:
        data = load_dataset("MBPP")
        data = concatenate_datasets([data[k] for k in data.keys()])
        samples = get_dict_list(args.input_file)
        if not args.start:
            args.start = 0
        if not args.end:
            args.end = len(samples)
        task_ids_range = list(range(args.start, args.end))
        if args.exclude_start is not None and args.exclude_end is not None:
            task_ids_range = list(range(args.start, args.exclude_start))+list(range(args.exclude_end, args.end))
        task_ids_range = set(task_ids_range)
        task_ids_range = task_ids_range.intersection(sanitized_task_ids)

        samples = [s for s in samples if s['task_id'] in task_ids_range]
        error_types = {}
        
        # ----
        eval_args = []
        for sample_dict in tqdm(samples, desc="vI, Evaluating and scoring..."):
            task_id = sample_dict["task_id"]
            if task_id not in task_ids_range:
                continue
            completion = sample_dict[args.completion_column_name]
            p_dict = {}
            r_dict = {}
            for shot, completion_i in completion.items():
                eval_args.append([data, task_id, completion_i])
        
        with NoDaemonPool(os.cpu_count()//2) as p:
            eval_res = p.map(eval_pool, eval_args)

        eval_idx = 0
        for sample_dict in tqdm(samples, desc="vII, Evaluating and scoring..."):
            task_id = sample_dict["task_id"]
            if task_id not in task_ids_range:
                continue
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
        #     if task_id not in task_ids_range:
        #         continue
        #     completion = sample_dict[args.completion_column_name]
        #     p_dict = {}
        #     r_dict = {}
        #     for shot, completion_i in completion.items():
        #         p, r = eval(data, task_id, completion_i)
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
        #error_types_i = Counter(error_types[shot])
        num_corr_results = compute_results(samples, shot)
        pass_at_k_results[shot] = compute_pass_at_ks(num_corr_results, ks)
        at_least_one_correct, _ = compute_at_least_one_pass_per_task(num_corr_results)
        pc_one_correct = at_least_one_correct / len(num_corr_results.keys())
        pass_at_k_results["% tasks with at least one passed completion"] = pc_one_correct
        print("\t", {key: round(val*100.0, 2) for key, val in pass_at_k_results[shot].items()})
        #print ("\t",error_types_i.most_common())

    if not os.path.exists(output_file):
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
