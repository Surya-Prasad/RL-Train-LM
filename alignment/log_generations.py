import torch
import wandb
import numpy as np
from tests.adapters import tokenize_prompt_and_output, get_response_log_probs

def get_rewards_and_lengths(responses, truths, token_ids, reward_func): 
    reward_total, reward_format, response_reward = list(), list(), list()
    resp_len, correct_resp_len, incorr_resp_len = list(), list(), list()

    for resp, truth, token_id in zip(responses, truths, token_ids): 
        rewards = reward_func(resp, truth)
        reward_total.append(rewards.get("reward", 0.0))
        reward_format.append(rewards.get("format", 0.0))
        response_reward.append(rewards.get("response", 0.0))

        tok_len = len(token_id)
        resp_len.append(tok_len)

        if rewards.get("reward", 0.0) == 1.0:
            correct_resp_len.append(tok_len)
        else:
            incorr_resp_len.append(tok_len)

    return {
        "rewards": {
            "total": reward_total,
            "format": reward_format,
            "response": response_reward
        },
        "lengths": {
            "response": resp_len,
            "correct": correct_resp_len,
            "incorrect": incorr_resp_len
        }
    }

def get_entropy(prompts, responses, policy_model, tokenizer, device = "cuda"): 
    entropy_list = list()

    for prompt, gen_text in zip(prompts, responses): 
        tknzd = tokenize_prompt_and_output(prompt, gen_text, tokenizer, device)
        if not tknzd: 
            entropy_list.append(0.0)
            continue

        input_ids = tknzd["input_ids"].to(device)
        labels = tknzd["labels"].to(device)
        mask = tknzd["response_mask"].to(device)

        with(torch.no_grad()): 
            dict_log_p = get_response_log_probs(policy_model, input_ids, labels, True)

        entropy_tensor = dict_log_p["token_entropy"]
        mask_sum = mask.sum().float()

        avg_entropy = entropy_tensor.masked_fill(~mask.bool(), 0.0).sum() / mask_sum if mask_sum > 0 else 0.0
        entropy_list.append(avg_entropy)

    return entropy_list

def reward_table_builder(prompts, responses, truths, rewards, entropy_list): 
    columnNames=[
        "Prompt", 
        "Response", 
        "Ground Truth", 
        "Reward: Format", 
        "Reward: Response", 
        "Reward: Total", 
        "Average Token Entropy", 
        "Average Response Length", 
        "Average Correct Response Length", 
        "Average Incorrect Response Length"
        ]
    wandb_table = wandb.Table(columnNames)

    for i in range(len(prompts)): 
        row = [
            prompts[i], 
            responses[i], 
            truths[i], 
            rewards["rewards"]["format"][i], 
            rewards["rewards"]["response"][i], 
            rewards["rewards"]["total"][i], 
            entropy_list[i], 
            rewards["lengths"]["response"][i], 
            rewards["lengths"]["correct"][i] if rewards["lengths"]["correct"] else 0.0, 
            rewards["lengths"]["incorrect"][i] if rewards["lengths"]["incorrect"] else 0.0
        ]

    metrics = {
        "eval/mean_total_reward" : np.mean(rewards["rewards"]["total"]), 
        "eval/mean_format_reward" : np.mean(rewards["rewards"]["format"]), 
        "eval/mean_response_reward" : np.mean(rewards["rewards"]["response"]), 
        "eval/mean_entropy" : np.mean(entropy_list), 
        "eval/mean_response_length" : np.mean(rewards["lengths"]["response"]), 
        "eval/mean_correct_response_length" : np.mean(rewards["lengths"]["correct"]), 
        "eval/mean_incorrect_response_length" : np.mean(rewards["lengths"]["incorrect"]), 
        "eval/generations_table": wandb_table
    }

    return metrics


def log_generations(vllm, policy_model, tokenizer, prompts, ground_truth, reward_func, sampling_params, step, device = "cuda"): 
    responses = vllm.generate(prompts, sampling_params)

    generated_resp = [resp.responses[0].text for resp in responses]
    token_ids_list = [resp.responses[0].token_ids for resp in responses]

    rewards_lengths = get_rewards_and_lengths(generated_resp, ground_truth, reward_func, token_ids_list)

    entropy_list = get_entropy(prompts, generated_resp, policy_model, tokenizer, device)

    metrics = reward_table_builder(prompts, generated_resp, ground_truth, rewards_lengths, entropy_list)

    wandb.log(metrics, step = step)

    return metrics