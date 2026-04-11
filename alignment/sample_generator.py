import random

def sample_prompts(dataset, batch_size):
    """Yields batches of prompts and their corresponding ground truths."""
    while True:
        sampled_batch = random.sample(dataset, batch_size)
        
        prompts = [
            example.get("problem", example.get("question", example.get("prompt", ""))) 
            for example in sampled_batch
        ]
        
        ground_truths = [
            example.get("solution", example.get("answer", example.get("expected_answer", ""))) 
            for example in sampled_batch
        ]
        
        yield prompts, ground_truths