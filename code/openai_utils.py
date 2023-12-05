# GPT-J
import torch 
import openai 
import time

def sample_code_from_gpt_models(args, prompt, model, tokenizer):
    output_strs = []
    num_samples = args.num_samples
    temperature = args.temperature
    debug = args.debug
    try:
        with torch.no_grad():
            input_ids = (
                torch.LongTensor(tokenizer.encode(prompt, verbose=False))
                .unsqueeze(0)
                .cuda()
            )
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,  # 0.2, 0.8
                max_length=1024 - len(input_ids),
                num_return_sequences=num_samples,
            )
            output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
            if debug:
                print(f"Input: {prompt}")
                print(f"Outputs: {output_strs}")
    except Exception as e:
        if (
            isinstance(e, UnboundLocalError)
            and str(e) == "local variable 'next_tokens' referenced before assignment"
        ):
            # See https://github.com/huggingface/transformers/issues/5118
            if debug:
                print("Problem text was > 2048 tokens, so cannot do generation")
                print(e)
        print(e)
    return output_strs

def initialize_openai(args):
    api_key = open(f"{args.openai_creds_dir}/openai_api_key.txt").read()
    # openai.organization = open(
    #     f"{args.openai_creds_dir}/openai_organization_id.txt"
    # ).read()
    openai.api_key = api_key


def sample_code_from_openai_model(args, prompt):
    output_strs = []
    start = time.time()

    arch_mapping = {
        "codex": "code-davinci-002",
        "gpt3": "text-davinci-001",
        "davinci-002": "text-davinci-002",
        "davinci-003": "text-davinci-003",
        "ada": "text-ada-001",
        "babbage": "text-babbage-001",
        "curie": "text-curie-001",
        "chatgpt": "gpt-3.5-turbo-0301"
    }
    engine_name = arch_mapping[args.arch]

    for i in range(args.num_samples):
        while time.time() - start < args.max_request_time:
            try:
                if args.arch == "chatgpt":
                    response = openai.ChatCompletion.create(
                        model=engine_name,
                        messages=prompt
                    )
                    output_strs += [
                        choice["message"] for choice in response["choices"]
                    ]
                else:
                    response = openai.Completion.create(
                        engine=engine_name,
                        prompt=prompt,
                        max_tokens=2048,
                        n=1,
                        temperature=args.temperature,
                    )
                    output_strs += [
                        choice["text"] for choice in response["choices"]
                    ]
                break
            except Exception as e:
                print(
                    f"Unexpected exception in generating solution. Sleeping again: {e}"
                )
                time.sleep(args.sleep_time)
    
    return output_strs

