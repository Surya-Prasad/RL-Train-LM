import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import numpy as np

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    prompt_len = len(prompt_strs)
    output_len = len(output_strs)

    if prompt_len < 1 or output_len < 1 or prompt_len != output_len: 
        print(f"run_tokenize_prompt_and_output: Length Mismatch, prompt: {prompt_len}, output: {output_len}")
        return {}

    # https://huggingface.co/docs/transformers/en/internal/tokenization_utils
    MAX_LEN = 2048 

    prompt_tknzd = tokenizer(
        prompt_strs, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=MAX_LEN
    )["input_ids"]

    output_tknzd = tokenizer(
        output_strs, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=MAX_LEN
    )["input_ids"]

    if not prompt_tknzd or not output_tknzd:
        print(f"run_tokenize_prompt_and_output: No List from Tokenizer - prompt_tknzd: {prompt_tknzd}, output_tknzd: {output_tknzd}")

    input_ids_list = list()
    response_mask_list = list()
    max_text_len = 0
    for i in range(prompt_len): 
        prompt_ids = prompt_tknzd[i]
        output_ids = output_tknzd[i] 

        text_ids = prompt_ids + output_ids

        mask = [False] * len(prompt_ids) + [True] * len(output_ids)

        input_ids_list.append(torch.tensor(text_ids))
        response_mask_list.append(torch.tensor(mask))

        max_text_len = max(max_text_len, len(text_ids))

    batch_size = len(input_ids_list)

    text_input_ids = torch.full((batch_size, max_text_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.full((batch_size, max_text_len), fill_value=False, dtype=torch.bool)

    for i in range(batch_size):
        seq_len = input_ids_list[i].shape[0]
        text_input_ids[i, :seq_len] = input_ids_list[i]
        response_mask[i, :seq_len] = response_mask_list[i]

    return {
        "input_ids": text_input_ids[:, :-1], 
        "labels": text_input_ids[:, 1:],
        "response_mask": response_mask[:, 1:]
    } 

def compute_entropy(logits):
    # print(logits.shape)
    # Suggestion from doc: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_p_x = logits - log_denom
    p_x = torch.exp(log_p_x)
    entropy = -1 * (p_x * log_p_x).sum(dim=-1)
    return entropy

def get_response_log_probs(model, input_ids, labels, return_token_entropy):
    # Suggestion from doc: Obtain logits with model(input_ids).logits
    logits = model(input_ids).logits.float()
    sigma_log_p = torch.nn.functional.log_softmax(logits, dim=-1)
    log_p = torch.gather(sigma_log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Suggestion from doc: You will want to use a numerically stable method to compute this, and are free to use methods from torch.nn.functional.
    entropy = compute_entropy(logits) if return_token_entropy else None

    return {
        "log_probs": log_p.float(),
        "token_entropy": entropy.float() if entropy is not None else None
    }

def masked_normalize(tensor, mask, dim, normalize_constant):
    bool_mask = mask.bool()

    masked_tensor = tensor.masked_fill(~bool_mask, 0)
    summed = masked_tensor.sum() if dim is None else masked_tensor.sum(dim=dim)
    return summed / normalize_constant

def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant):
    batch_size = policy_log_probs.size(0)
    negative_log_likelihood = -1 * policy_log_probs
    masked_negative_log_likelihood = masked_normalize(negative_log_likelihood, response_mask, None, normalize_constant)

    loss = masked_negative_log_likelihood / batch_size

    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {
        "unscaled_loss": masked_negative_log_likelihood.detach(),
        "scaled_loss": scaled_loss.detach(),
    }

    return scaled_loss, metadata