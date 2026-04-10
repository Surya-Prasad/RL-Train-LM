import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.sft_modules import (
    tokenize_prompt_and_output, 
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step
)
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.vllm_ops import init_vllm,load_policy_into_vllm_instance

def grpo_train_loop(args):
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B")
    
    vllm_engine = init_vllm("models/Qwen2.5-Math-1.5B", "cuda", seed=42069)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.0, 
        betas=(0.9, 0.95)
    )
    
    wandb.init(project="math-grpo")

    for step in range(args.n_grpo_steps):
        print(f"--- Starting GRPO Step {step} ---")
        
        load_policy_into_vllm_instance(model, vllm_engine)
        
        # 2. Sample a batch of prompts (rollout_batch_size // group_size prompts)
        batch_prompts = sample_prompts(dataset, args.n_prompts_per_rollout_batch)
        
        # 3. Format prompts with the r1_zero template
        # The prompt asks the model to stop at </answer>
        formatted_prompts = [r1_zero_template.format(question=p) for p in batch_prompts]
        
        # 4. Generate G responses per prompt using vLLM
        # Ensure to repeat the prompts G times and pass them to vLLM
        repeated_prompts = [p for p in formatted_prompts for _ in range(args.group_size)]
        vllm_outputs = vllm_engine.generate(repeated_prompts, sampling_params)
        rollout_responses = [out.outputs[0].text for out in vllm_outputs]
        
        # ==========================================
        # PHASE 2: REWARDS & ADVANTAGES
        # ==========================================
        # 1. Match responses with repeated ground truths
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(args.group_size)]
        
        # 2. Calculate group-normalized advantages using your helper
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization
        )
        
        # Log your rewards!
        wandb.log({"train/mean_reward": reward_metadata["reward_mean"], "train_step": step})