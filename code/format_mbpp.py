from datasets import load_dataset, concatenate_datasets
from file_utils import write_jsonl

def get_entry_point(code, test_list):
    func_sig = code.split("def ")[1]
    lparen_idx = func_sig.index("(")
    entrypoint = func_sig[:lparen_idx]
    if entrypoint == "__init__":
        assert_statement = test_list[0]
        if "assert (" in assert_statement:
            assert_prefix = "assert ("
        elif "assert " in assert_statement:
            assert_prefix = "assert "
        assert_statement = assert_statement[len(assert_prefix) :]
        lparen_idx = assert_statement.index("(")
        entrypoint = assert_statement[:lparen_idx]
    return entrypoint

def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False
    
def format_test(entrypoint, test_list):
    test_str = "def check(candidate):\n"

    # use pytest.approx() for float results
    if is_float(test_list[0].split("==")[-1]):
        answer_float = True
    else:
        answer_float = False 
    test_str = "from pytest import approx\n\n" + test_str
    for i in range(len(test_list)):
        split = test_list[i].split("==")
        func_sig =  " ".join("==".join(split[:-1]).split(' ')[1:])
        input_args = func_sig.split(entrypoint)[-1].replace("'", '"').strip()
        if answer_float:
            assert_stmt = "assert x == approx(y)"
        else:
            assert_stmt = "assert x == y"
        split[-1] = f"; y = {split[-1].strip()}; {assert_stmt}, 'Input: {input_args} Output: '+str(x)+' Expected: '+str(y)"
        test_list[i] =  "x = "+func_sig+ split[-1]

    for test in test_list:
        test_str += f"\t{test}\n"
    test_str += "\n"

    if entrypoint != "check":
        test_str = test_str.replace(entrypoint, "candidate")
    else:
        test_str = test_str.replace(f"assert {entrypoint}", "candidate")
    return test_str


if __name__ == "__main__":
    data = load_dataset("mbpp")
    data = concatenate_datasets([data[k] for k in data.keys()])

    test_list = []
    entrypoints = []
    for i in range(len(data)):
        task_id = data["task_id"][i]
        code = data["code"][i]
        test = data["test_list"][i]
        entrypoint = get_entry_point(code, test)
        print (entrypoint)
        test_str = format_test(entrypoint, test)
        test_list.append(test_str)
        entrypoints.append(entrypoint)
    data = data.add_column("test_formatted", test_list)
    data = data.add_column("entrypoint", entrypoints)
    write_jsonl(data, 'MBPP/mbpp_formatted.json')

