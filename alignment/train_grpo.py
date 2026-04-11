import json
import random
import typer
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from alignment.sft_modules import (
    tokenize_prompt_and_output, 
    get_response_log_probs,
)
from alignment.grpo_modules import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.vllm_ops import init_vllm, load_policy_into_vllm_instance

from alignment.sample_generator import sample_prompts

app = typer.Typer()

def load_math_dataset(file_path="data/MATH/train.jsonl"):
    """Loads the MATH dataset from a jsonl file."""
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

@app.command()
def grpo_train_loop(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    seed: int = 42069
):
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vllm_engine = init_vllm(model_id, "cuda", seed=seed, gpu_memory_utilization=0.4)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.0, 
        betas=(0.9, 0.95)
    )
    
    wandb.init(project="math-grpo")

    dataset = load_math_dataset("data/MATH/train.jsonl")
    prompt_iterator = iter(sample_prompts(dataset, n_prompts_per_rollout_batch))

    with open("alignment/prompts/r1_zero.prompt", "r") as f:
        r1_zero_template = f.read()

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        n=group_size, 
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    for step in range(n_grpo_steps):
        print(f"--- Starting GRPO Step {step} ---")
        
        load_policy_into_vllm_instance(model, vllm_engine)
        
        batch_prompts, batch_ground_truths = next(prompt_iterator)
        
        formatted_prompts = [r1_zero_template.format(question=p) for p in batch_prompts]
        
        vllm_outputs = vllm_engine.generate(formatted_prompts, sampling_params)
        
        rollout_responses = []
        repeated_ground_truths = []
        for out, gt in zip(vllm_outputs, batch_ground_truths):
            for request_output in out.outputs:
                rollout_responses.append(request_output.text)
                repeated_ground_truths.append(gt)

        repeated_prompts = [p for p in formatted_prompts for _ in range(group_size)]
        
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        
        wandb.log({
            "train_step": step,
            "train/mean_reward": reward_metadata.get("reward_mean", raw_rewards.float().mean().item()),
            "train/mean_advantage": advantages.mean().item()
        })

        model.train()
        
        tokenized_data = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized_data["input_ids"].to("cuda")
        labels = tokenized_data["labels"].to("cuda")
        response_mask = tokenized_data["response_mask"].to("cuda")
        
        advantages = advantages.to("cuda").unsqueeze(-1)
        if raw_rewards is not None:
            raw_rewards = raw_rewards.to("cuda").unsqueeze(-1)

        old_log_probs = None
        if loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
                old_log_probs = old_log_probs_dict["log_probs"]

        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(rollout_batch_size) if epochs_per_rollout_batch > 1 else torch.arange(rollout_batch_size)
            
            for i in range(0, rollout_batch_size, train_batch_size):
                batch_indices = indices[i:i+train_batch_size]
                optimizer.zero_grad()
                
                for j in range(0, train_batch_size, micro_train_batch_size):
                    micro_indices = batch_indices[j:j+micro_train_batch_size]
                    
                    mb_input_ids = input_ids[micro_indices]
                    mb_labels = labels[micro_indices]
                    mb_response_mask = response_mask[micro_indices]
                    mb_advantages = advantages[micro_indices]
                    mb_old_log_probs = old_log_probs[micro_indices] if old_log_probs is not None else None
                    mb_raw_rewards = raw_rewards[micro_indices] if raw_rewards is not None else None
                    
                    policy_log_probs_dict = get_response_log_probs(model, mb_input_ids, mb_labels, return_token_entropy=True)
                    policy_log_probs = policy_log_probs_dict["log_probs"]
                    token_entropy = policy_log_probs_dict["token_entropy"]
                    
                    scaled_loss, log_grpo_step = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=mb_response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange
                    )
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                wandb.log({
                    "train_step": step,
                    "train/loss": log_grpo_step.get("unscaled_loss", scaled_loss * gradient_accumulation_steps).item(),
                    "train/token_entropy": token_entropy.mean().item() if token_entropy is not None else 0.0,
                    "train/clip_fraction": log_grpo_step.get("clip_fraction", 0.0)
                })

if __name__ == "__main__":
    app()