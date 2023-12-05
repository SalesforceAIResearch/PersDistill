from eval_humaneval import *
import argparse
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_file',
        type=str,
        default='json_results.json', 
        help="")
    parser.add_argument(
        '--k',
        type=str,
        default='5,10,20', 
        help="")
    parser.add_argument(
        '--clean_data',
        type=str,
        default='humaneval_clean_all', 
        help="")    

    return parser.parse_args()

def load_data(args):
    samples = [json.loads(x) for x in open(args.output_file).readlines()]
    shots = set(samples[0]['passed'].keys())
    return samples, shots

def filter_samples(args, samples):
    clean_test_data = load_from_disk(args.clean_data)
    clean_task_ids = set([example['task_id'] for example in clean_test_data])
    clean_samples = [sample for sample in samples if sample['task_id'] in clean_task_ids]
    return clean_samples

def print_results(samples, shots):
    ks = [int(elem) for elem in args.k.split(",")]
    pass_at_k_results = {}

    for shot in sorted(shots):
        print ("\nShot ", shot)
        num_corr_results = compute_results(samples, shot)
        pass_at_k_results[shot] = compute_pass_at_ks(num_corr_results, ks)
        print("\t", {key: round(val*100.0, 2) for key, val in pass_at_k_results[shot].items()})

if __name__ == '__main__':
    # load samples
    args = parse_args()
    samples, shots = load_data(args)
    clean_samples = filter_samples(args, samples)
    
    print("Original data:")
    print_results(samples, shots)
    print("Clean data:")
    print_results(clean_samples, shots)