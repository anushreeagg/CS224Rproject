import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import random

def main():

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    

    dataset = load_dataset("Asap7772/cog_behav_all_strategies", split="train")
    

    print(f"Dataset keys: {list(dataset[0].keys())}")
    if len(dataset) > 100:
        sample_indices = random.sample(range(len(dataset)), 100)
    else:
        sample_indices = range(len(dataset))
    
    input_lengths = []
    output_lengths = []
    
    for idx in sample_indices:
        example = dataset[idx]
        

        input_tokens = tokenizer(example["query"], return_length=True)
        output_tokens = tokenizer(example["completion"], return_length=True)
        
        input_lengths.append(input_tokens["length"])
        output_lengths.append(output_tokens["length"])
    

    input_min = min(input_lengths)
    input_max = max(input_lengths)
    input_mean = np.mean(input_lengths)
    
    output_min = min(output_lengths)
    output_max = max(output_lengths)
    output_mean = np.mean(output_lengths)
    

    print("\nToken length statistics for 100 samples:")
    print(f"Input tokens  -> Min: {input_min}, Max: {input_max}, Mean: {input_mean:.2f}")
    print(f"Output tokens -> Min: {output_min}, Max: {output_max}, Mean: {output_mean:.2f}")

    print("\nSample example:")
    sample_idx = sample_indices[0]
    print(f"Input: {dataset[sample_idx]['query'][:100]}...")
    print(f"Target: {dataset[sample_idx]['completion'][:100]}...")

if __name__ == "__main__":
    main() 