import re
from datasets import concatenate_datasets, load_dataset
import ast 
import tqdm 
from file_utils import write_jsonl

def check_in_prompt(output, input_args, prompt):
    if output in prompt:
        return True 
    elif input_args in prompt:
        return True 
    else:
        try:
            input_args = ast.literal_eval(input_args)
        except:
            print ("error with \n", input_args)
            input_args = [x.strip() for x in input_args.split(",")]
        if type(input_args)!=tuple:
            input_args = [input_args]
        prompt_lines = prompt.split("\n")
        for line in prompt_lines:
            if all([str(arg) in line for arg in input_args]):
                return True 
        return False 



def format_test(tests, prompt=None, mode=None):
    if mode not in ['seen', 'all']:
        raise ValueError("mode for format_test should be either seen or all")
    entry_point = "candidate"
    tests = [line for line in tests.split("\n") if 'assert' in line and entry_point in line]
    delims = ['==', ' is ', ' not ', ' < ', ' > ', '!=']
    test_lines = []
    for test in tests: 
        test_delim = None  
        for delim in delims:
            if delim in test:
                test_delim = delim 
                break 
        if test_delim is None:
            test = test +" == True"
            test_delim = "=="
        elif test_delim==" not ":
            test = test.replace(" not ", " ")+" == False"
            test_delim = "=="
        elif test_delim ==" is ":
            test = test.replace(" is ", " == ")
            test_delim = "=="
        if test_delim:
            test = test.split(test_delim)
            left = test[0]
            right = test[1]
            left = re.sub("assert ", "", left)
            expected = ""
            output = ""
            if entry_point in left:
                try:
                    input_args = re.search("candidate\((.+?)\)", left).group(1)
                except:
                    input_args = left.split("candidate")[-1].strip()
                output = left
                expected = right
            elif entry_point in right:
                raise Exception("entry point cannot be in right")
            output = output.strip()
            expected = expected.strip()
            if output.startswith("(") and expected.endswith(")"):
                output = output[1:]
                expected = expected[:-1]
            if not (input_args.startswith("(") and input_args.endswith(")")):
                input_args = "("+input_args+")"
            if mode=="seen":
                in_prompt = check_in_prompt(output, input_args, prompt)
                if not in_prompt:
                    continue
            delim_str = "'"+test_delim+"'"
            test = "check_assert("+delim_str+", "+input_args+", "+output+", "+expected+")"
            test_lines.append(test)
    return test_lines

def get_seen_tests(data, task_id):
    formatted_tests = []
    idx = data["task_id"].index(task_id)
    prompt = data['prompt'][idx]
    entry_point = data['entry_point'][idx]
    test = None
    prompt = prompt.replace('==>', '==').replace('=>', '==').replace('->', '==').replace("# =>", "==").replace("➞", "==").replace("should return", "==").replace("returns", "==")
    prompt = "\n".join([line.split(" #")[0] for line in prompt.split("\n")])
    if '>>>' in prompt:
        prompt = "\n".join(re.sub('\n', ' == ', prompt).split(">>>")[1:])
    if 'Output:' in prompt and 'Input:' in prompt:
        prompt = re.sub('\n[ \t]*Output:', " == ", prompt)
    for line in prompt.split("\n"):
        line = line.replace("true", "True").replace("false", "False")
        regex = re.findall('For (.+) the output should be (.+)', line)
        if len(regex)>0:
            left = regex[0][0].strip()
            right = regex[0][1].strip()
            if entry_point not in line:
                left = entry_point+"("+left+")"
            test = "assert "+left+" == "+right 
            formatted_tests.append(test)
        regex = re.findall('For (.+) the result should be (.+)', line)
        if len(regex)>0:
            left = regex[0][0].strip()
            right = regex[0][1].strip()
            if entry_point not in line:
                left = entry_point+"("+left+")"
            test = "assert "+left+" == "+right 
            formatted_tests.append(test)
        elif 'Input: ' in line and '==' in line:
            left = line.split("==")[0]
            right = line.split("==")[1]
            left = left.replace('Input:','')
            if entry_point not in left:
                left = entry_point+"("+left+")"
            test = "assert "+left+" == "+right 
            formatted_tests.append(test)
        elif entry_point in line:
            line = line[re.search(entry_point, line).start():]
            line = line.replace(" = ",  " == ")
            if '==' in line:
                left = line.split("==")[0]
                right = line.split("==")[1]
                test = "assert "+left+" == "+right 
                formatted_tests.append(test)
        elif '==' in line:
            left = line.split("==")[0]
            right = line.split("==")[1]
            if entry_point not in left:
                left = entry_point+"("+left+")"
            test = "assert "+left+" == "+right 
            formatted_tests.append(test)
        elif ' = ' in line:
            left = line.split(" = ")[0]
            right = line.split(" = ")[1]
            if entry_point not in left:
                left = entry_point+"("+left+")"
            test = "assert "+left+" == "+right 
            formatted_tests.append(test)

    clean_tests = []
    for test in formatted_tests:
        test = re.sub(' +',' ', test)
        test = test.replace(entry_point, "candidate")
        try:
            ast.parse(test)
            clean_tests.append(test)
        except:
            print ("ERROR:: ", test)

    if len(clean_tests)==0:        
        clean_tests = format_test(data['test'][idx], prompt, mode="seen")
    else:
        clean_tests = "\n".join(clean_tests)
        clean_tests = format_test(clean_tests, mode="all")

    return clean_tests

  


def get_all_tests(data, task_id, assert_function=None, mode="infer"):
    idx = data["task_id"].index(task_id)
    test = data['test'][idx]
    prompt = data['prompt'][idx]
    if mode=="eval":
        return test 
    elif mode=="seen":
        clean_tests = get_seen_tests(data, task_id)
    elif mode=="all":
        clean_tests = format_test(test, mode="all")
    test_str = assert_function+"\n\n"+"def check(candidate):\n"
    for test in clean_tests:
        test = test.strip()
        test_str += f"\t{test}\n"
    return test_str 
    

if __name__=="__main__":
    assert_func = open('templates/assert_template.txt').read()
    data = load_dataset("openai_humaneval")
    data = concatenate_datasets([data[k] for k in data.keys()])
    test_seen = []
    test_all = []
    print (data)
    for i in range(len(data)):
        task_id = data["task_id"][i]
        test_seen.append(get_all_tests(data, task_id, assert_function=assert_func, mode="seen"))
        test_all.append(get_all_tests(data, task_id, assert_function=assert_func, mode="all"))
    data = data.add_column("test_seen", test_seen)
    data = data.add_column("test_all", test_all)
    write_jsonl(data, 'HumanEval/humaneval_formatted.json')
    