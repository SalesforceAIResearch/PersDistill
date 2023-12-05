import argparse
import logging
import re
from file_utils import write_jsonl
import json

def segment_data(args):
    segment_data = [json.loads(line) for line in open(args.train_file).readlines()]
    chunk_size = 1 + len(segment_data) // args.segment_k
    segments = []
    for j in range(0, len(segment_data), chunk_size):
        segments.append(segment_data[j : j + chunk_size])
    
    output_file_template = args.output_dir + '/finetuning_data_chatgpt_rectification_segment_{}.jsonl'
    for i, data in enumerate(segments):
        write_jsonl(data, output_file_template.format(i))

def parse_args(input_args):
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning prompts from model-generated refinements. Also generate FT prompts for those same task IDs from the original MBPP dataset using gold code."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to file containing train data (refinement only)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output segmented data",
    )

    parser.add_argument(
        "--segment_k",
        type=int,
        default=5,
        help="Number of segments",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args(None)
    segment_data(args)


if __name__ == "__main__":
    main()
