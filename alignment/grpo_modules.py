import torch
import numpy as np

def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std): 
    rewards_list = list()

    for resp, truth in zip(rollout_responses, repeated_ground_truths):
        rewards_dict = reward_fn(resp, truth)
        rewards_list.append(rewards_dict.get("reward", 0.0))

    rewards = torch.tensor(rewards_list, dtype=torch.float32)

    num_groups = len(rewards) // group_size
    group_rewards = rewards.view(num_groups, group_size)

    group_means = group_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std: 
        group_std = group_rewards.std(dim=1, unbiased = False, keepdim=True)
        advantages = (group_rewards - group_means) / (group_std + advantage_eps)
    else:
        advantages = group_rewards - group_means

    advantages = advantages.view(-1, 1)

    metadata = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_max": rewards.max().item(),
        "reward_min": rewards.min().item(),
    }
    return advantages, rewards, metadata

