import argparse
import json
import logging
import os
import pprint
import re
import numpy as np 
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from eval_mbpp import eval, get_error_text
from file_utils import write_jsonl
from openai_utils import initialize_openai, sample_code_from_openai_model
from chat_utils import end_token, Conversation, sample_decode, greedy_decode, post_process_pred
import re 
import torch
from hf_credentials import auth_token
import time
import gc 

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def format_prompt(task_id, text, tests, sample_code, num_prompts):
    # Create prompt from scratch
    prompt = f'"""\n{text}\n\n'
    if num_prompts > 0:
        for i in range(num_prompts):
            example = tests[i].split("assert ")[-1].replace("==", "=")
            prompt += f">>> Example: {example}\n"

    # Add code prefix
    fn_name = tests[0].split("assert ")[-1].split("(")[0]
    fn_search = re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", sample_code)
    if fn_search is None:
            return None 
    code_prefix = sample_code[: fn_search.end()]
    prompt = f'{prompt}"""\n\n{code_prefix}\n'
    return prompt

def get_clean_completion(completion, prompt):
    if completion.startswith(prompt):
        completion = completion[len(prompt):]
    prompt_head = prompt.split('"""')[-1].split("def ")[0]
    func_sig = "def "+prompt.split('"""')[-1].split("def ")[1]
    if func_sig.strip() not in completion:
        completion = func_sig+"\n"+completion
    prompt_last_line = prompt.strip('\n').split('\n')[-1]
    fn_name = prompt_last_line[re.search(f"def ", prompt_last_line).start():].split(' ')[1].split('(')[0]
    try:
        completion = completion[re.search(f"def [ ]*{fn_name}[ ]*\(.*\)[ ]*:", completion).start():]
        eof_m = re.search(r'\n[A-Za-z#"]+?', completion)
        if eof_m is not None:
            completion = completion[: eof_m.start() + 1]
    except:
        return completion
    completion = prompt_head+"\n"+completion.strip()
    return completion 


def fill_prompt_template(text, to_replace):
    for k, v in to_replace.items():
        text = text.replace(k, v)
    return text 

def sample_code_from_starcoder(args, data, task_id, prompt, model, tokenizer, num_samples=None, shot=1):
    completions = {}
    if not num_samples:
        num_samples = args.num_samples 
    conv = Conversation()
    conv.append_message('user', prompt)
    conv.append_message('assistant', None)
    input_prompt = conv.get_prompt()
    input_ids = tokenizer(
        input_prompt, truncation=True, max_length=1024, return_tensors="pt"
    ).input_ids.cuda()
    if args.temperature == 0.0:
        args.num_samples = 1
    try:
        # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
        if args.temperature > 0:
            if num_samples > 1:
                tokens = []
                for count in range(int(args.num_samples)):
                    tokens_i = sample_decode(
                        input_ids,
                        model,
                        tokenizer,
                        max_length=1024,
                        temperature=args.temperature,
                        top_p=0.95,
                        stop_words=[end_token]
                    )
                    tokens.extend(tokens_i)
            else:
                tokens = sample_decode(
                    input_ids,
                    model,
                    tokenizer,
                    max_length=1024,
                    top_p=0.95,
                    temperature=args.temperature,
                    stop_words=[end_token]
                )
        else:
            tokens = greedy_decode(
                    input_ids,
                    model,
                    tokenizer,
                    max_length=1024,
                    stop_words=[end_token]
                )
    except RuntimeError as e:
        logging.error(f"Could not sample from model: {e}")
    assert len(tokens) == num_samples
    completions = {shot: [post_process_pred(text, end_token) if end_token in text else text for text in tokens]}
    # print(completions[1][0])
    # import pdb;pdb.set_trace()

    
    if shot == 1 and args.refine:

        #print ("\n\n==============================\nPROMPT:")
        #print (prompt)
        
        if args.feedback_type == 'nl':
            prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
        elif args.feedback_type == 'auto':
            prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'

        ft_refinement_prompt = open(prompt_filepath).read()
        #print ("\nCOMPLETION:")
        #print (completions[shot][0])
        eval_time = []
        for completion in completions[shot]:
            shot = 1 
            while shot <= args.num_refine:
                shot += 1 
                if shot not in completions:
                    completions[shot] = []
                to_replace = {}
                to_replace["<<TASK>>"] = "" 
                to_replace["<<CODE>>"] = completion
                header = prompt.split('"""')[-1]+"\n"
                to_replace["<<HEADER>>"] = header 
                completion_to_eval = get_clean_completion(prompt=prompt, completion=completion)
                start_t = time.time()
                pass_value, return_value = eval(data, task_id=task_id, completion=completion_to_eval)
                eval_time.append(time.time() - start_t)
                if pass_value:
                    completions[shot].append(completion)
                    continue

                if args.feedback_type == "auto":
                    error_text = get_error_text(return_value=return_value)
                    #print ('error text:\n'+error_text)
                    to_replace["<<ERROR>>"] = error_text
                elif args.feedback_type == "nl":
                    to_replace["<<FEEDBACK>>"] = "Mistakes in the code"

                refinement_prompt = fill_prompt_template(ft_refinement_prompt, to_replace=to_replace)
                refinement = sample_code_from_starcoder(args, data, task_id, refinement_prompt, model, tokenizer, num_samples=1, shot=shot)
                refinement = refinement[shot][0]
                refinement = header + "\n" + refinement
                if refinement.lower().strip() == "none" or len(refinement.split("\n"))<3:
                    completion = completion
                else:
                    completion = refinement
                completions[shot].append(completion)
        #print ("\nREFINEMENT:")
        #print (completions[args.num_refine][0])
        #print ("\n-----------------------------------------------------------\n\n")
    #print(f"Eval time: {np.array(eval_time).mean()} * {len(completions[shot])} seconds")
    return completion

def sample_code_from_codegen(args, data, task_id, prompt, model, tokenizer, num_samples=None, shot=1):
    completions = {}
    start_t = time.time()
    if not num_samples:
        num_samples = args.num_samples 
    inputs = tokenizer(
        prompt, truncation=True, max_length=1024, return_tensors="pt"
    ).to('cuda')
    if args.temperature == 0.0:
        args.num_samples = 1
    try:
        # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
        if args.temperature > 0:
            if num_samples > 1:
                tokens = []
                inf_batch_size = 2
                for start_idx in range(0, num_samples, inf_batch_size):
                    inf_bs_ = min(start_idx+inf_batch_size, num_samples) - start_idx
                # for count in range(int(args.num_samples/2)):
                    tokens_i = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=inf_bs_,
                        max_length=inputs.input_ids.shape[1] + 1024,
                        temperature=args.temperature,
                        top_p=0.95,
                        use_cache=True,
                    )
                    #print(f"Line 203: {time.time() - start_t:.2f} seconds {inf_bs_}")
                    start_t = time.time()
                    tokens.extend(tokens_i)
            else:
                tokens = model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=1,
                    max_length=inputs.input_ids.shape[1] + 1024,
                    temperature=args.temperature,
                    top_p=0.95,
                    use_cache=True,
                )
                #print(f"Line 216: {time.time() - start_t:.2f} seconds")
                start_t = time.time()
        else:
            tokens = model.generate(
                **inputs,
                num_return_sequences=1,
                max_length=inputs.input_ids.shape[1] + 1024,
                use_cache=True,
            )
    except RuntimeError as e:
        logging.error(f"Could not sample from model: {e}")
    assert len(tokens) == num_samples
    tokens = tokenizer.batch_decode(tokens)

    # completions = {shot: [post_process_pred(text, "<|endoftext|>") for text in tokens]}
    completions = {shot: [text[: text.find("<|endoftext|>")] if "<|endoftext|>" in text else text for text in tokens]}
    # eval_time = []
    if shot == 1 and args.refine:

        #print ("\n\n==============================\nPROMPT:")
        #print (prompt)
        
        if args.feedback_type == 'nl':
            prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
        elif args.feedback_type == 'auto':
            prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'
        elif args.feedback_type == 'none':
            prompt_filepath = 'prompts/ft_rectify_instruction.txt'

        ft_refinement_prompt = open(prompt_filepath).read()
        #print ("\nCOMPLETION:")
        #print (completions[shot][0])
        for completion in completions[shot]:
            shot = 1 
            while shot <= args.num_refine:
                shot += 1 
                if shot not in completions:
                    completions[shot] = []
                
                completion_to_eval = get_clean_completion(prompt=prompt, completion=completion)
                pass_value, return_value = eval(data, task_id=task_id, completion=completion_to_eval)
                # eval_time.append(time.time() - start_t)
                if pass_value:
                    completions[shot].append(completion)
                    continue

                to_replace = {}
                to_replace["<<TASK>>"] = "" 
                to_replace["<<CODE>>"] = completion_to_eval
                header = prompt.split('"""')[-1]+"\n"
                to_replace["<<HEADER>>"] = header 
                if args.feedback_type == "auto":
                    error_text = get_error_text(return_value=return_value)
                    #print ('error text:\n'+error_text)
                    to_replace["<<ERROR>>"] = error_text
                elif args.feedback_type == "nl":
                    to_replace["<<FEEDBACK>>"] = "Mistakes in the code"

                refinement_prompt = fill_prompt_template(ft_refinement_prompt, to_replace=to_replace)
                # print(f"Line 270: {time.time() - start_t:.2f} seconds")
                start_t = time.time()
                refinement = sample_code_from_codegen(args, data, task_id, refinement_prompt, model, tokenizer, num_samples=1, shot=shot)
                refinement = refinement[shot][0]
                refinement = refinement[len(refinement_prompt):]
                
                if  ("code is correct" in refinement.lower() or refinement.lower().strip() == "none") and len(refinement.split("\n"))<2:
                    for shot_ in range(shot, args.num_refine+2):
                        if shot_ not in completions:
                            completions[shot_] = []
                        completions[shot_].append(completion)
                    break 
                else:
                    refinement = header + "\n" + refinement
                    completions[shot].append(refinement)
        #print ("\nREFINEMENT:")
        #print (completions[args.num_refine][0])
        #print ("\n-----------------------------------------------------------\n\n")
    # mean_eval_time = np.array(eval_time).mean()
    # print(f"Eval time: {mean_eval_time} * {len(eval_time)} seconds")
    return completions

def sample_code_from_codegen_v2(args, data, task_id, prompt, model, tokenizer, entrypoint=None, num_samples=None, shot=1):
    def generate_k_v2(inputs, inf_batch_size=1):
        ''' assume len(inputs.input_ids) >= inf_batch_size, split wrt input
        '''
        
        tokens = []
        num_inputs = len(inputs.input_ids)
        batch_len = inputs.input_ids.size(1)
        true_len_per_input = [batch_len - inputs.attention_mask[i].tolist().index(1) for i in range(num_inputs)]
        true_len_per_input_sorted = sorted(true_len_per_input)
        # sort inputs by true length (bucket batcher)
        bucket_ids = sorted([i for i in range(num_inputs)], key=lambda x: true_len_per_input[x])
        inverse_bucket_ids = np.argsort(bucket_ids)
        start_idx = 0

        sorted_inputs_ids = inputs.input_ids[bucket_ids]
        sorted_attention_mask = inputs.attention_mask[bucket_ids]
        # long_flag = False # true when moving to long inputs
        while start_idx < num_inputs: # not done yet
            end_idx = start_idx
            for inc_idx in range(start_idx+1, min(start_idx+inf_batch_size, num_inputs)):
                if true_len_per_input_sorted[inc_idx] > 700: # sample with input length > threshold, will not be batched
                    break
                end_idx = inc_idx
            
            batch_input_ids = sorted_inputs_ids[start_idx:(end_idx+1)]
            batch_attention_mask = sorted_attention_mask[start_idx:(end_idx+1)]
            padding_prefix_idx = min([batch_attention_mask[i].tolist().index(1) for i in range(batch_attention_mask.size(0))])
            batch_input_ids = batch_input_ids[:, padding_prefix_idx:]
            batch_attention_mask = batch_attention_mask[:, padding_prefix_idx:]
            
            try:
                tokens_i = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    do_sample=True,
                    num_return_sequences=1,
                    max_length=inputs.input_ids.shape[1] + args.max_gen_len,
                    temperature=args.temperature,
                    top_p=0.95,
                    use_cache=True,
                )
            except Exception as e:
                logging.error(f"Could not sample from model: {e}")
                import pdb;pdb.set_trace()

            tokens.extend(tokens_i)
            start_idx = end_idx + 1
        # import pdb;pdb.set_trace()
        return [tokens[idx] for idx in inverse_bucket_ids]
        
        # for start_idx in range(0, num_inputs, inf_batch_size):
        #     batch_input_ids = inputs.input_ids[start_idx:start_idx+inf_batch_size]
        #     batch_attention_mask = inputs.attention_mask[start_idx:start_idx+inf_batch_size]
        #     tokens_i = model.generate(
        #         input_ids=batch_input_ids,
        #         attention_mask=batch_attention_mask,
        #         do_sample=True,
        #         num_return_sequences=1,
        #         max_length=inputs.input_ids.shape[1] + args.max_gen_len,
        #         temperature=args.temperature,
        #         top_p=0.95,
        #         use_cache=True,
        #     )
        #     tokens.extend(tokens_i)
        # return tokens

    def generate_k(inputs, num_samples, inf_batch_size=1):
        ''' assume len(inputs)==1, num_samples >= inf_batch_size, split computation wrt. outputs
        '''
        tokens = []
        for start_idx in range(0, num_samples, inf_batch_size):
            inf_bs_ = min(start_idx+inf_batch_size, num_samples) - start_idx
            tokens_i = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=inf_bs_,
                max_length=inputs.input_ids.shape[1] + args.max_gen_len,
                temperature=args.temperature,
                top_p=0.95,
                use_cache=True,
            )
            tokens.extend(tokens_i)
        return tokens

    def generate_greedy(inputs):
        tokens = model.generate(
                **inputs,
                num_return_sequences=1,
                max_length=inputs.input_ids.shape[1] + args.max_gen_len,
                use_cache=True,
            )
        return tokens

    def generate_(inputs, num_samples):
        if args.arch in ["codegen-16B", "starcoder"]:
            first_inf_batch_size = 2
            second_inf_batch_size = 1
        else:
            first_inf_batch_size = 3
            second_inf_batch_size = 2
        try:
            # Note: max_length is max length of input IDs, and max_length_sample is max length for completion (not including input IDs)
            if args.temperature > 0:
                try:
                    if num_samples == 1:
                        tokens = generate_k_v2(inputs, inf_batch_size=second_inf_batch_size)
                    else:
                        assert inputs.input_ids.size(0) == 1
                        tokens = generate_k(inputs, num_samples, inf_batch_size=first_inf_batch_size)
                except Exception as e:
                    logging.error(f"Could not sample from model: {e}, re-trying with batch_size 1 with num_samples {num_samples}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    if num_samples == 1:
                        tokens = generate_k_v2(inputs, inf_batch_size=1)
                    else:
                        tokens = generate_k(inputs, num_samples, inf_batch_size=1)
            else:
                tokens = generate_greedy(inputs)
        except RuntimeError as e:
            logging.error(f"Could not sample from model: {e}")
        
        # assert len(tokens) == inputs.input_ids.size(0) * num_samples, f"generated {len(tokens)}, not {inputs.input_ids.size(0)} * {num_samples}"
        tokens = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        completions = [text[: text.find("<|endoftext|>")] if "<|endoftext|>" in text else text for text in tokens]
        return completions

    completions = {}
    start_t = time.time()
    if not num_samples:
        num_samples = args.num_samples 

    inputs = tokenizer(
        prompt, truncation=True, max_length=args.max_context_len, return_tensors="pt"
    ).to('cuda')

    completions_ = generate_(inputs, num_samples)
    completions = {1: completions_}

    if shot == 1 and args.refine:

        #print ("\n\n==============================\nPROMPT:")
        #print (prompt)
        
        if args.feedback_type == 'nl':
            prompt_filepath = 'prompts/ft_feedback_rectify_instruction.txt'
        elif args.feedback_type == 'auto':
            prompt_filepath = 'prompts/ft_error_rectify_instruction.txt'
        elif args.feedback_type == 'none':
            prompt_filepath = 'prompts/ft_rectify_instruction.txt'

        ft_refinement_prompt = open(prompt_filepath).read()
        
        prev_shot = 1
        def generate_more_shot(prev_shot):
            cur_shot = prev_shot + 1
            cur_query_prompts = {} # {$shot : {idx: prompt}}
            cur_completion = {i:None for i in range(len(completions[prev_shot]))} # {idx: completion}
            prev_completions = completions[prev_shot]
            for idx, completion in enumerate(prev_completions):
                
                completion_to_eval = get_clean_completion(prompt=prompt, completion=completion)
                pass_value, return_value = eval(data, task_id=task_id, completion=completion_to_eval)

                if pass_value:
                    cur_completion[idx] = completion
                    continue

                to_replace = {}
                to_replace["<<TASK>>"] = "" 
                to_replace["<<CODE>>"] = completion_to_eval
                header = prompt.split('"""')[-1]+"\n"
                to_replace["<<HEADER>>"] = header
                if args.feedback_type == "auto":
                    error_text = get_error_text(return_value=return_value)
                    to_replace["<<ERROR>>"] = error_text
                elif args.feedback_type == "nl":
                    to_replace["<<FEEDBACK>>"] = "Mistakes in the code"

                refinement_prompt = fill_prompt_template(ft_refinement_prompt, to_replace=to_replace)
                cur_query_prompts[idx] = refinement_prompt
            
            query_prompts_cur_shot = list(cur_query_prompts.values())
            if query_prompts_cur_shot:
                inputs = tokenizer(
                    query_prompts_cur_shot, truncation=True, max_length=args.max_context_len, return_tensors="pt", padding=True
                ).to('cuda')
                # import pdb;pdb.set_trace()
                completions_ = generate_(inputs, num_samples=1)
                for completion, idx in zip(completions_, cur_query_prompts.keys()):
                    refinement_prompt = cur_query_prompts[idx]
                    refinement = completion[len(refinement_prompt):]
                    refinement = header + "\n" + refinement
                    if refinement.lower().strip() == "none" or len(refinement.split("\n"))<3:
                        completion = prev_completions[idx]
                    else:
                        completion = refinement
                    cur_completion[idx] = completion
            return list(cur_completion.values())

        for shot in range(2, 2+args.num_refine):
            # print(f'shot {shot}')
            completions[shot] = generate_more_shot(prev_shot=shot-1)

    return completions


def generate_feedback_for_problems(args, code_filepath):
    outputs = []
    initialize_openai(args)

    args.arch = "chatgpt"

    data = load_dataset("MBPP")
    data = concatenate_datasets([data[k] for k in data.keys()])

    instruction_turns = []

    if args.feedback_type == 'nl':
        prompt_filepath = 'prompts/chatgpt_feedback_instruction.txt'
    elif args.feedback_type == 'auto':
        prompt_filepath = 'prompts/chatgpt_error_instruction.txt'
    turn = []
    for line in open(prompt_filepath).readlines():
        if line.startswith("=") and len(set(line.strip()))==1:
            instruction_turns.append("\n".join(turn))
            turn = []
        else:
            turn.append(line.strip())

    code_data = [json.loads(line) for line in open(code_filepath).readlines()]
    for data_i in code_data:
        prompt = data_i['prompt']
        completion = data_i['completion']
        task_id = data_i['task_id']
        pass_value, return_value = eval(data, task_id, completion=completion)
    
        if pass_value:
            continue 
        error_text = None 
        prompt_header = prompt.split('"""')[-1]
        prompt_task = prompt[:-len(prompt_header)]
        to_replace = {"<<TASK>>": prompt_task, "<<HEADER>>": prompt_header, "<<CODE>>": completion}
        if len(instruction_turns)>1:
            turn1_prompt = instruction_turns[0]
            turn1_prompt = fill_prompt_template(turn1_prompt, to_replace)
            turn1_prompt = [{"role": "user", "content": turn1_prompt}]
            #print ("\nCHATGPT PROMPT FOR GOLD:")
            #print (turn1_prompt[0]['content'])
            if args.get_openai_gold:
                gold = sample_code_from_openai_model(args, turn1_prompt)[0]['content']
            else:
                gold = data_i['gold']
            turn1_prompt.append({"role": "assistant", "content": gold})
            turn2_prompt = instruction_turns[1]
        else:
            turn1_prompt = None 
            turn2_prompt = instruction_turns[0]
            gold = None 
            
        if not pass_value:
            error_text = get_error_text(return_value)
            to_replace["<<ERROR>>"] = error_text
        else:
            continue 
        turn2_prompt = fill_prompt_template(turn2_prompt, to_replace)

        if turn1_prompt is None:
            turn2_prompt = [{"role": "user", "content": turn2_prompt}]
        else:
            turn2_prompt = turn1_prompt+[{"role": "user", "content": turn2_prompt}]

        feedback = sample_code_from_openai_model(args, turn2_prompt)

        d = {
                "task_id": data_i['task_id'],
                "prompt": prompt,
                "original_completion": completion,
                "chatgpt_output": feedback[0]['content']
        }
        if args.get_openai_gold and gold:
            d["chatgpt_gold"] = gold

        if error_text:
            d["original_error_text"] = error_text

        outputs.append(d)

        # print ("ORIGINAL COMPLETION:")
        # print (completion)
        # print ("\nCHATGPT PROMPT:")
        # print (turn2_prompt[-1]['content'])
        # print ("\nCHATGPT RECTIFICATION")
        # print (feedback[0]['content'])
        # if gold:
        #     print ("\nCHATGPT GOLD:")
        #     print (d["chatgpt_gold"])
        # print ("\n------------------------------------------\n\n")
    return outputs


def generate_code_for_problems(args):
    data = load_dataset("MBPP")
    data = concatenate_datasets([data[k] for k in data.keys()])

    output = []
    if args.arch in ["gpt3", "codex", "chatgpt"]:
        initialize_openai(args)
        prompt_format = open("prompts/chatgpt_instruction.txt").read()
        gen_fn = sample_code_from_openai_model
            
    elif args.arch.startswith("codegen"): 
        tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.arch}-mono")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if args.arch == "codegen-6B":

            if args.model_path is None:
                model = AutoModelForCausalLM.from_pretrained(
                    f"Salesforce/{args.arch}-mono",
                    device_map="auto", pad_token_id=tokenizer.eos_token_id)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, device_map="auto", pad_token_id=tokenizer.eos_token_id)
            
            gen_fn = sample_code_from_codegen_v2
                
            gen_fn = sample_code_from_codegen_v2
                
        elif args.arch == "codegen-16B":

            if args.model_path is None:
                model = AutoModelForCausalLM.from_pretrained(
                    f"Salesforce/{args.arch}-mono", torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    device_map="auto", pad_token_id=tokenizer.eos_token_id)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, 
                    device_map="auto", pad_token_id=tokenizer.eos_token_id)
            
            gen_fn = sample_code_from_codegen
    
    elif args.arch in ['starchat-alpha']:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        gen_fn = sample_code_from_codegen
    elif args.arch in ['starcoder']:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, use_auth_token=auth_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=auth_token)
        gen_fn = sample_code_from_codegen
        tokenizer.padding_side = "left"


    if not args.use_mbpp_full:

        mbpp_orig = load_dataset('mbpp', 'sanitized')
        mbpp_orig = concatenate_datasets([mbpp_orig['test'], mbpp_orig['validation'], mbpp_orig['prompt']])
                                        
        sanitized_task_ids = set(mbpp_orig['task_id'])
        assert sanitized_task_ids == set([2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 77, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 113, 115, 116, 117, 118, 119, 120, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 137, 138, 139, 140, 141, 142, 143, 145, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 222, 223, 224, 226, 227, 228, 229, 230, 232, 233, 234, 235, 237, 238, 239, 240, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 290, 291, 292, 293, 294, 295, 296, 297, 299, 300, 301, 304, 305, 306, 307, 308, 309, 310, 311, 312, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 468, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 554, 555, 556, 557, 558, 559, 560, 562, 563, 564, 565, 566, 567, 568, 569, 572, 573, 574, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600])

        task_ids_range = [data['task_id'][i] for i in range(len(data)) if data['task_id'][i] in sanitized_task_ids]
    else:
        task_ids_range = [data['task_id'][i] for i in range(len(data))]
        
    if args.start is not None and args.end is not None:
        task_ids_range = [x for x in task_ids_range if x >=args.start and x <args.end]

    if args.proc_id is None: 
        indices_filtered = [i for i in range(len(data)) if data["task_id"][i] in task_ids_range]
        data_filtered = data.select(indices_filtered)
        print ("Total tasks: ", len(task_ids_range))
    elif args.proc_id is not None:
        task_ids_range = np.array_split(task_ids_range, args.num_procs)[args.proc_id]
        print ("Total tasks: ", len(task_ids_range))
        indices_filtered = [i for i in range(len(data)) if data["task_id"][i] in task_ids_range]
        data_filtered = data.select(indices_filtered)


    for i in tqdm(range(len(data_filtered))):
        data_i = data_filtered[i]
        prompt = format_prompt(
            data_i["task_id"],
            data_i["text"],
            data_i["test_list"],
            data_i["code"],
            args.num_demos,
        )
        gold = data_i["code"]
        if prompt is None:
            continue
        if args.arch == "chatgpt":
            prompt_header = prompt.split('"""')[-1]
            prompt_task = prompt[:-len(prompt_header)]
            to_replace = {"<<TASK>>": prompt_task, "<<HEADER>>": prompt_header}
            prompt = fill_prompt_template(prompt_format, to_replace)
            cg_prompt = [{"role": "user", "content": prompt}]
            completions = gen_fn(args, prompt=cg_prompt)
        else:
            task_id = data_i["task_id"]
            completions = gen_fn(args, data, task_id=task_id, prompt=prompt, model=model, tokenizer=tokenizer)
        shot = 1 
        for i,completion in enumerate(completions[shot]):
            if completion is None:
                continue 
            if True or args.arch in ['starchat-alpha']:
                output.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt,
                        "gold": gold,
                        "completion": {s: completions[s][i] for s in completions}
                    }
                )
            else:
                output.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt,
                        "gold": gold,
                        "completion": {s: get_clean_completion(completions[s][i], prompt) for s in completions}
                    }
                )

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained model to generate Python code for the MBPP benchmark."
    )
    parser.add_argument(
        "--arch",
        default="gptj",
        choices=[
            "gptj",
            "codex",
            "gpt3",
            "chatgpt",
            "codegen-16B",
            "codegen-6B",
            "davinci-002",
            "davinci-003",
            "ada",
            "babbage",
            "curie",
            "starchat-alpha",
            "starcoder",
        ],
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Directory to load model checkpoint from. If None, will load a pre-trained "
        "CodeGen model using the --arch argument instead.",
    )

    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="Which MBPP split to use. In datasets v1.16.1, MBPP only has the split 'test'.",
    )
    parser.add_argument(
        "-s", "--start", default=None, type=int, help="Task ID to start with."
    )
    parser.add_argument(
        "-e", "--end", default=None, type=int, help="Task ID to end with (exclusive)."
    )

    parser.add_argument(
        "-p", "--num-procs", default=None, type=int, help="Number of processes to run (=number of gpus)"
    )

    parser.add_argument(
        "-i", "--proc-id", default=None, type=int, help="Process id (in [0, num_proc-1])"
    )

    parser.add_argument(
        "-r",
        "--num-refine",
        default=1,
        type=int,
        help="Number of times to refine outpur during inference.",
    )

    parser.add_argument(
        "--num-demos",
        default=1,
        type=int,
        help="Number of examples from doc string to show in prompt.",
    )

    parser.add_argument(
        "--feedback-type",
        type=str,
        help="type of feedback needed: nl or auto"
    )

    parser.add_argument(
        "--max-gen-len",
        default=1024,
        type=int,
        help="max number of tokens to generate",
    )

    parser.add_argument(
        "--max-context-len",
        default=1024,
        type=int,
        help="max number of tokens to input",
    )
    
    parser.add_argument(
        "--max-request-time",
        type=int,
        default=80,
        help="Max. time to wait for a successful GPT-3 request.",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=10,
        help="Time to sleep (in seconds) between each GPT-3 call.",
    )
    parser.add_argument(
        "--openai-creds-dir",
        type=str,
        default='openai_creds',
        help="Directory where OpenAI API credentials are stored. Assumes the presence of "
        "openai_api_key.txt and openai_organization_id.txt files.",
    )

    parser.add_argument(
        "--use-mbpp-full",
        action="store_true",
        help="Whether to use full mbpp or mbpp sanitized only (false for mbpp_sanitized)"

    )
    parser.add_argument(
        "--get-openai-feedback",
        action="store_true",
        help="Whether to get feedback from openai GPT models on the generated code"
    )

    parser.add_argument(
        "--get-openai-gold",
        action="store_true",
        help="whether to get gold data from openai GPT models or not"
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether to further refine the code generation or not"
    )

    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    if args.start is not None and args.end is not None:
        start_end = f"{args.start}-{args.end}"
    else:
        start_end = ""
    if args.proc_id is not None:
        args.output_file_suffix = args.output_file_suffix+"_"+str(args.proc_id)
    output_filepath = os.path.join(
        args.output_dir,
        f"samples_{args.split}_{args.arch}_{args.num_samples}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl",
    )
    if not os.path.exists(output_filepath):
        completions = generate_code_for_problems(args)
        write_jsonl(completions, output_filepath)
    else:
        print ("Not re-running, Already found code output existing in: ", output_filepath)

    if args.get_openai_feedback:
        feedback_filepath = os.path.join(
            args.output_dir,
            f"chatgpt_feedback_{args.split}_{args.arch}_{args.num_samples}shot_temp{args.temperature}_{start_end}{args.output_file_suffix}.jsonl"
        )
        if not os.path.exists(feedback_filepath):
            feedbacks = generate_feedback_for_problems(args, output_filepath)
            write_jsonl(feedbacks, feedback_filepath)
        else:
            print ("Not re-runnning, Already found openai feedback existing in: ", feedback_filepath)

if __name__ == "__main__":
    main(parse_args())