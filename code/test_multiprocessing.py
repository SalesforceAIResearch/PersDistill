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

def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

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
    print(all_src)
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

def main(completion, tests):
    for test in tests:
        #print (test)
        try:
            p, r = eval_code_wrapper(completion, test)
            print(p, r)
        except Exception as e:
            with io.StringIO() as f:
                traceback.print_exception(*sys.exc_info(), file=f)
                r = f.getvalue()
            p = False
            print(f"Caught exception from eval_code: {e}\n{r}")

if __name__ == '__main__':
    completion = "\n\n\ndef sort_list(list):\n\n\n    \"\"\"\n    Sort the elements of a list in ascending order.\n\n    Args:\n    list: list of integers\n\n    Returns:\n    list: list of integers sorted in ascending order\n    \"\"\"\n    list.sort()\n    return list"
    tests = [{'input':'sort_list([4,1,2,3])', 'output':'[1,2,3,4]'}]
    main(completion, tests)
