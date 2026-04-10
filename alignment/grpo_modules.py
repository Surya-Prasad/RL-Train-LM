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
        group_stds = group_rewards.std(dim=1, unbiased=True, keepdim=True)
        advantages = (group_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages = group_rewards - group_means

    advantages = advantages.flatten()

    log_rewards = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_max": rewards.max().item(),
        "reward_min": rewards.min().item(),
    }
    return advantages, rewards, log_rewards

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return -1 * (raw_rewards_or_advantages * policy_log_probs)

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    ratio = torch.exp(policy_log_probs - old_log_probs)

    objective_unclipped = ratio * advantages

    g_adv = advantages + (cliprange * torch.abs(advantages))
    loss = -torch.min(objective_unclipped, g_adv)

    is_clipped = (g_adv < objective_unclipped).float()

    log_gClip = {
        "clip_fraction": is_clipped.mean(),
    }

    return loss, log_gClip


def compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange): 
    log_pgc = dict()

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards required for no_baseline")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages required for reinforce_with_baseline")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        if advantages is None: 
            raise ValueError("advantages required for grpo_clip")
        
        elif old_log_probs is None:
            raise ValueError("old_log_probs required for grpo_clip")

        elif cliprange is None:
            raise ValueError("cliprange required for grpo_clip")

        loss, log_pgc = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    return loss, log_pgc

