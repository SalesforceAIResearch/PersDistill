import json

path_to_1gold = 'models/CCG_ALP_16B_1_new/finetuning_data_chatgpt_gold.jsonl'
path_to_previous_2gold = 'models/finetuning_data_chatgpt_only_two_gold_cleaned_pool.jsonl'

d1 = [json.loads(x) for x in open(path_to_1gold).readlines()]
d2 = [json.loads(x) for x in open(path_to_previous_2gold).readlines()]

d2_task_ids = set([d['task_id'] for d in d2])
d1_task_ids = set([d['task_id'] for d in d1])

still_required_gold_task_ids = d1_task_ids - d2_task_ids
print(f'd1 set: {len(d1_task_ids)}, d2 set: {len(d2_task_ids)}')
print(f'number of tasks which still requires gold: {len(still_required_gold_task_ids)}')
# import pdb;pdb.set_trace()