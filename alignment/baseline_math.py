import json
import torch
from typing import Callable, List
from vllm import LLM, SamplingParams
from alignment.drgrpo_grader import r1_zero_reward_fn

def load_math_validation_data(filepath: str):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn,
    prompts,
    ground_truths,
    eval_sampling_params: SamplingParams,
    output_path: str = "zero_shot_baseline_results.json"
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results_to_save = []
    counts = {"category_1": 0, "category_2": 0, "category_3": 0}
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]
        
        rewards = reward_fn(generated_text, gold_answer)
        f_reward = rewards.get("format_reward", 0.0)
        a_reward = rewards.get("answer_reward", 0.0)
        
        if f_reward == 1.0 and a_reward == 1.0:
            counts["category_1"] += 1
        elif f_reward == 1.0 and a_reward == 0.0:
            counts["category_2"] += 1
        else: 
            counts["category_3"] += 1
            
        results_to_save.append({
            "problem": prompts[i],
            "solution": gold_answer,
            "generated": generated_text,
            "rewards": rewards
        })

    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
        
    total = len(prompts)
    print(f"Results: {counts}")
    print(f"Overall Accuracy: {(counts['category_1']/total)*100:.2f}%")

if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-Math-1.5B"
    val_path = "data/MATH/validation.jsonl"
    prompt_tpl_path = "alignment/prompts/r1_zero.prompt"
    
    with open(prompt_tpl_path, "r") as f:
        r1_zero_prompt = f.read()

    val_examples = load_math_validation_data(val_path)
    prompts = [r1_zero_prompt.format(question=ex["problem"]) for ex in val_examples]
    ground_truths = [ex["solution"] for ex in val_examples]

    llm = LLM(model=model_path, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)