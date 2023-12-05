from datasets import load_dataset, concatenate_datasets
import os 
import json 
import re 
from file_utils import write_jsonl
import io 
import timeout_decorator
import traceback
from multiprocessing import Process, Queue
import sys 
import ast
from pprint import pprint

@timeout_decorator.timeout(3)
def eval_code(q, src, test):
    all_src = f"{src}\n{test}\n"
    try:
        exec(all_src, {})
    except Exception:
        with io.StringIO() as f:
            traceback.print_exception(*sys.exc_info(), file=f)
            q.put((False, f.getvalue()))
        return
    q.put((True, None))

@timeout_decorator.timeout(5)
def eval_code_wrapper(src, test):
    queue = Queue()
    p = Process(target=eval_code, args=(queue, src, test))
    p.start()
    p.join(3)
    if p.is_alive():
        p.kill()
    if not queue.empty():
        return queue.get()
    else:
        return False, f"Exit code: {p.exitcode}"
    

def extract_input_from_unittest(input_file, output_file):
    codeT_humaneval_data = [json.loads(x) for x in open(input_file).readlines()]
    humaneval = load_dataset("HumanEval")
    humaneval = concatenate_datasets([humaneval[k] for k in humaneval.keys()])
    entrypoint_to_taskid_map = {}
    for i in range(len(humaneval)):
        task_id = humaneval[i]['task_id']
        entrypoint = humaneval[i]['entry_point']
        entrypoint_to_taskid_map[entrypoint] = task_id

    num_instances_notest = 0
    entrypoint_to_inputargs_map = {}
    for i in range(len(codeT_humaneval_data)):
        prompt = codeT_humaneval_data[i]['prompt']
        samples = codeT_humaneval_data[i]['samples']
        prompt = prompt.split("# check the correctness of")
        code = prompt[0]
        entrypoint = prompt[1].split("\n")[0].strip()

        task_id = entrypoint_to_taskid_map[entrypoint]
        input_args_list = []
        for sample in samples:
            for line in sample.split("\n"):
                if line.strip().startswith("assert ") and entrypoint in line:
                    unittest = line 
                    #print ("\n", line)
                    #pprint(ast.dump(ast.parse(line)))
                    pattern = entrypoint+"\((.+?)\)"
                    try:
                        input_args = re.search(pattern, unittest).group(1)
                    except:
                        input_args = unittest.split(entrypoint)[-1].strip()
                    input_args = entrypoint+"(" + input_args+ ")"
                    try:
                        p, r = eval_code_wrapper(code, input_args)
                    except Exception as e:
                        with io.StringIO() as f:
                            traceback.print_exception(*sys.exc_info(), file=f)
                            r = f.getvalue()
                    if p is True:
                        input_args_list.append(input_args)
                    else:
                        print ("Failed!!!!!   ",input_args)
        
        input_args_list = list(set(input_args_list))
        entrypoint_to_inputargs_map[task_id] = input_args_list
    
    input_args = []    
    for i in range(len(humaneval)):
        task_id = humaneval[i]['task_id']
        if task_id in entrypoint_to_inputargs_map:
            input_args.append(entrypoint_to_inputargs_map[task_id])
        else:
            input_args.append([])
            num_instances_notest += 1

    print ("num_instances_notest ",num_instances_notest)
    humaneval = humaneval.add_column("test_input_args", input_args)
    write_jsonl(humaneval, output_file)
                    


if __name__=="__main__":
    codeT_data_dir = 'CodeT/generated_data'
    codeT_humaneval_filename = 'HumanEval_codegen16B_temp0.8_topp0.95_num100_max300_test_case.jsonl'
    input_file = os.path.join(codeT_data_dir, codeT_humaneval_filename)
    output_file = 'HumanEval_CodeT/codeT_humaneval_data.json'


    extract_input_from_unittest(input_file, output_file)
