import sys
import os
import torch
import random
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_samples(model_path, num_samples=5, max_new_tokens=100):
    """
    Generate sample outputs from the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        num_samples: Number of samples to generate
        max_new_tokens: Maximum number of new tokens to generate
    """
    #load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    #fine-tuned model weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    #test data
    test_dataset = load_dataset("cais/mmlu_countdown", split="test")
    
    #random samples
    if len(test_dataset) > num_samples:
        sample_indices = random.sample(range(len(test_dataset)), num_samples)
    else:
        sample_indices = range(len(test_dataset))
    
    
    os.makedirs("outputs", exist_ok=True)
    

    generation_results = []
    
    for idx in sample_indices:
        example = test_dataset[idx]
        input_text = example["input"]
        
        #input print:
        print(f"\n\n{'='*50}")
        print(f"SAMPLE {idx}")
        print(f"{'='*50}")
        print(f"INPUT:\n{input_text}")
        

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGENERATED OUTPUT:\n{generated_text}")
        

        ground_truth = example['target']
        print(f"\nGROUND TRUTH:\n{ground_truth}")
        

        generation_results.append({
            "prompt": input_text,
            "generation": generated_text,
            "ground_truth": ground_truth
        })
    
    #Save results to JSON
    with open("outputs/generations.json", "w") as f:
        json.dump(generation_results, f, indent=2)
    
    print(f"\nGeneration results saved to outputs/generations.json")

if __name__ == "__main__":
  
    model_path = "../final_model" #saif you'd prolly need to change this to finetuned model path
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please provide a valid path.")
        print("Using default model for demonstration purposes.")
        model_path = "Qwen/Qwen1.5-0.5B"  
    
    generate_samples(model_path) 