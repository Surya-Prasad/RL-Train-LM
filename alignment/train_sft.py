import os
import json
import torch
import wandb
import typer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step
)
from sft_modules import log_generations, init_vllm, load_policy_into_vllm_instance
# Assuming your reward function is accessible
from alignment.drgrpo_grader import r1_zero_reward_fn

def load_sft_data(filepath: str, max_examples: int = None):
    """Loads the prompt-response pairs from the SFT jsonl file."""
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    if max_examples:
        data = data[:max_examples]
    return [d["prompt"] for d in data], [d["response"] for d in data]

def main(
    data_path: str = "data/sft.jsonl",
    val_data_path: str = "data/MATH/validation.jsonl",
    model_id: str = "models/Qwen2.5-Math-1.5B",
    output_dir: str = "models/Qwen2.5-Math-1.5B-SFT",
    max_examples: int = None,
    epochs: int = 3,
    train_batch_size: int = 16,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    eval_every_n_steps: int = 50,
):
    # --- 1. Setup & WandB Initialization ---
    wandb.init(project="reasoning-sft", config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    device = "cuda" if torch.cuda.is_available() else "mps"
    micro_batch_size = train_batch_size // gradient_accumulation_steps

    # --- 2. Load Model, Tokenizer, and vLLM ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Using bfloat16 and flash_attention_2 as required
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)
    policy_model.train()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # Initialize vLLM for validation evaluations
    vllm_engine = init_vllm(model_id=model_id, device=device, seed=42069)
    from vllm import SamplingParams
    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # --- 3. Load Data ---
    train_prompts, train_responses = load_sft_data(data_path, max_examples)
    val_prompts, val_targets = load_sft_data(val_data_path, max_examples=1024) # Eval on 1024 examples [cite: 828-829]
    print(f"Loaded {len(train_prompts)} training examples.")

    # --- 4. Training Loop ---
    train_step = 0
    eval_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Simple batching mechanism
        for i in tqdm(range(0, len(train_prompts), micro_batch_size)):
            batch_prompts = train_prompts[i : i + micro_batch_size]
            batch_responses = train_responses[i : i + micro_batch_size]
            
            if not batch_prompts:
                continue

            # Tokenize prompt and response 
            tknzd = run_tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
            input_ids = tknzd["input_ids"].to(device)
            labels = tknzd["labels"].to(device)
            response_mask = tknzd["response_mask"].to(device)

            # Forward pass & log probs 
            log_prob_dict = run_get_response_log_probs(
                model=policy_model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True
            )

            # Compute SFT Loss and Backprop 
            loss, metadata = run_sft_microbatch_train_step(
                policy_log_probs=log_prob_dict["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0
            )

            # Optimizer Step (Accounting for Gradient Accumulation) 
            if (i // micro_batch_size + 1) % gradient_accumulation_steps == 0 or (i + micro_batch_size) >= len(train_prompts):
                # Gradient Clipping 
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                wandb.log({
                    "train/loss": metadata["unscaled_loss"].item(),
                    "train_step": train_step
                })
                train_step += 1

                # --- 5. Periodic Evaluation ---
                if train_step % eval_every_n_steps == 0:
                    print(f"\nRunning evaluation at step {train_step}...")
                    policy_model.eval()
                    
                    # Sync PyTorch weights to vLLM 
                    load_policy_into_vllm_instance(policy_model, vllm_engine)
                    
                    log_generations(
                        vllm_model=vllm_engine,
                        policy_model=policy_model,
                        tokenizer=tokenizer,
                        prompts=val_prompts,
                        ground_truths=val_targets,
                        reward_fn=r1_zero_reward_fn,
                        sampling_params=eval_sampling_params,
                        step=eval_step,
                        device=device
                    )
                    
                    eval_step += 1
                    policy_model.train()

    # --- 6. Save Final Model --- 
    print(f"Saving model to {output_dir}")
    policy_model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    wandb.finish()

if __name__ == "__main__":
    typer.run(main)