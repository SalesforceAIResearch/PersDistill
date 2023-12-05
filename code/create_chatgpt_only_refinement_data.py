import argparse 
import pprint 
import json 
from file_utils import write_jsonl

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="directory containing relevant data"
    )

    parser.add_argument(
        "--gold-file",
        type=str,
        default='finetuning_data_gold_.jsonl',
        help="File containing the chatgpt refinement data",
    )

    parser.add_argument(
        "--chatgpt-refinement-file",
        type=str, 
        default='finetuning_data_chatgpt_rectification_.jsonl',
        help="File containing the original gold data"
    )

    parser.add_argument(
        "--output-file",
        type=str, 
        default='finetuning_data_chatgpt_only.jsonl',
        help="Output file containing chatgpt only refinement data"
    )
    parser.add_argument(
        "--output-gold-file",
        type=str, 
        default='finetuning_data_chatgpt_gold.jsonl',
        help="Output file containing gold data corresponding to chatgpt only refinement data"
    )
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    gold_data = [json.loads(x) for x in open(args.gold_file).readlines()]
    refinement_data = [json.loads(x) for x in open(args.chatgpt_refinement_file).readlines()]

    gold_data = {d['task_id']: d for d in gold_data}
    refinement_taskids = set([d['task_id'] for d in refinement_data])
    refinement_only_data = refinement_data
    chatgpt_gold_data = []
    for k,v in gold_data.items():
        if k in refinement_taskids:
            refinement_only_data.append(v)
            chatgpt_gold_data.append(v)
    
    if args.output_file:
        write_jsonl(refinement_only_data, args.output_file)
    if args.output_gold_file:
        write_jsonl(chatgpt_gold_data, args.output_gold_file)


if __name__=="__main__":
    main(parse_args())
