from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Iterator, Callable
import torch
import transformers
import re

system_token = "<|system|>"
user_token = "<|user|>"
assistant_token = "<|assistant|>"
end_token = "<|end|>"

def prepare_dialogue(example):
    system_msg = "Below is a dialogue between a human and an AI assistant called StarChat."
    prompt = system_token + "\n" + system_msg + end_token + "\n"
    for message in example["messages"]:
        if message["role"] == "user":
            prompt += user_token + "\n" + message["content"] + end_token + "\n"
        else:
            prompt += assistant_token + "\n" + message["content"] + end_token + "\n"
    return prompt

@dataclass
class Conversation:

    """A class that keeps all conversation history."""
    # All messages
    messages: List[List[str]] = field(default_factory=lambda: [])
    # Two roles
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    # role tags
    role_tags: Dict[str, str] = field(default_factory=lambda: {'user':user_token, 'assistant': assistant_token, 'system': system_token})
    # line seperator
    line_sep: str = '\n'
    # eos seperator
    eos: str = end_token
    # System prompts
    system: str = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    
    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        ret = self.role_tags['system'] + self.line_sep + self.system + self.eos + self.line_sep
        for role, message in self.messages:
            if message:
                ret += self.role_tags[role] + self.line_sep + message + self.eos + self.line_sep
            else:
                ret += self.role_tags[role] + self.line_sep
        return ret


    def append_message(self, role: str, message: str): 
        """Append a new message."""
        self.messages.append([role, message])



def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        if top_k is not None:
           probs_sort1, _ = torch.topk(probs_sort, top_k)
           min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
           probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)
        # print(text)
        if any([x in text for x in stop_words]):
            return [text]
    return [text]

def greedy_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    stop_words: list,
    max_length: int,
) -> str:
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        if any([x in text for x in stop_words]):
            return text
    return text


def post_process_pred(pred_text: str, end_token: str) -> str:
    """ parse the predicted codes to clean python function code """
    # firstly, if ```python {code_block}``` exists, use the code_block
    template = "```(?:python\n)(.*?)```"
    match = re.search(template, pred_text, re.S)
    if match:
        return match.group(1)
    
    # secondely, if ```import xxx \n return xxx \n``` exists, use these code
    template = f"import .*?def .*\s\s+return.*?(?:\n|{re.escape(end_token)})"
    match = re.search(template, pred_text, re.S)
    if match:
        return match.group(0).replace(end_token, '')

    # thirdly, if ```def xxx \n return xxx \n``` exists, use these code
    template = f"def .*\s\s+return.*?(?:\n|{re.escape(end_token)})"
    match = re.search(template, pred_text, re.S)
    if match:
        return match.group(0).replace(end_token, '')
    
    # post-process failed
    print('####### post processing failed #######')
    return pred_text.replace(end_token, '')
