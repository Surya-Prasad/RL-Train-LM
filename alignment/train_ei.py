import torch
import json
import random
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.sft_modules import sft_microbatch_train_step, tokenize_prompt_and_output
from alignment.drgrpo_grader import r1_zero_reward_fn

def run_expert_iteration(
    model_id="Qwen/Qwen2.5-Math-1.5B",
    n_ei_steps=5, 
    G=8, 
    batch_size=512,
    epochs=1
):

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).to(device)


    with open("data/MATH/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    for step in range(n_ei_steps):
        print(f"--- Expert Iteration Step {step+1}/{n_ei_steps} ---")
        
        llm = LLM(model=model_id, gpu_memory_utilization=0.8)
        sampling_params = SamplingParams(
            temperature=1.0, 
            max_tokens=1024, 
            min_tokens=4, 
            n=G, 
            stop=["</answer>"] 
        )
        
        questions_batch = random.sample(train_data, batch_size)
        prompts = [q["problem"] for q in questions_batch]
        
        print(f"Generating {batch_size * G} rollouts...")
        outputs = llm.generate(prompts, sampling_params)
        
        # 3. FILTERING PHASE (Verified Rewards)
        # Create a temporary SFT dataset from correct traces 
        d_sft = []
        for i, output in enumerate(outputs):
            gold_answer = questions_batch[i]["solution"]
            for completion in output.outputs:
                reward_dict = r1_zero_reward_fn(completion.text, gold_answer)
                if reward_dict["reward"] == 1.0: 
                    d_sft.append({"prompt": prompts[i], "response": completion.text})
        
        print(f"Found {len(d_sft)} correct reasoning traces.")
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            random.shuffle(d_sft)
            for i in range(0, len(d_sft), 4):
                batch = d_sft[i:i+4]
                if not batch: continue
                
                toks = tokenize_prompt_and_output(
                    [b["prompt"] for b in batch], 
                    [b["response"] for b in batch], 
                    tokenizer
                )
                
                logits = model(toks["input_ids"].to(device)).logits
                
                optimizer.step()
                optimizer.zero_grad()
        
        model.save_pretrained(f"Qwen/expert_step_{step}")
        print(f"Step {step} complete.")

if __name__ == "__main__":
    run_expert_iteration()