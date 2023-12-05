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
        default='finetuning_data_gold__.jsonl',
        help="File containing the original gold data",
    )

    parser.add_argument(
        "--chatgpt-refinement-file",
        type=str, 
        default='finetuning_data_chatgpt_rectification__.jsonl',
        help="File containing the chatgpt refinement data"
    )

    parser.add_argument(
        "--output-file",
        type=str, 
        default='finetuning_data_chatgpt_newgold_.jsonl',
        help="Output file containing chatgpt refinement data as new gold data"
    )

    parser.add_argument(
        "--output-rect-file",
        type=str, 
        default='finetuning_data_chatgpt_newgold_rectification_only_.jsonl',
        help="Output file containing chatgpt refinement data as new gold data"
    )
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    gold_data = [json.loads(x) for x in open(args.gold_file).readlines()]
    refinement_data = [json.loads(x) for x in open(args.chatgpt_refinement_file).readlines()]

    gold_data = {d['task_id']: d for d in gold_data}
    
    newgold_data = []
    newgold_rect_data = []
    for v in refinement_data:
        k = v['task_id']
        v['finetuning_prompt'] = gold_data[k]['finetuning_prompt']
        newgold_data.append(v)
        newgold_data.append(gold_data[k])
        newgold_rect_data.append(v)

    #write_jsonl(newgold_data, args.output_file)
    write_jsonl(newgold_rect_data, args.output_rect_file)



if __name__=="__main__":
    main(parse_args())
