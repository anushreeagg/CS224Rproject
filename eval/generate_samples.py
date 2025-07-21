import sys
import os
import torch
import random
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def transform_countdown(json_file_path):
    data = []
    with open(json_file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            numbers_str = "[" + ", ".join(map(str, entry["num"])) + "]" 
            target_str = str(entry["target"])
            query = (f"A conversation between User and Assistant. The user asks a question, "
                     f"and the Assistant solves it. The assistant first thinks about the "
                     f"reasoning process in the mind and then provides the user with the answer.\n"
                     f"User: Using the numbers {numbers_str}, create an equation that equals "
                     f"{target_str}. You can use basic arithmetic operations (+, -, *, /) and each "
                     f"number can only be used once. Show your work in <think> </think> tags. And "
                     f"return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
                     f"Assistant: Let me solve this step by step.")
            data.append({
                "query": query,
                "completion": ""
            })
    dataset = Dataset.from_list(data)
    return dataset


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_samples(model_path, max_new_tokens=1028, k=None, save=True, dataset="default", model=None):
    """
    Generate sample outputs from the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        num_samples: Number of samples to generate
        max_new_tokens: Maximum number of new tokens to generate
    """
    # If no model arg is passed, use the path retrieve it 
    if not model: 
        #load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        #fine-tuned model weights
        model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    #test data
    if dataset == "default":
        test_dataset = transform_countdown("data/countdown.json")
    else:
        test_dataset = dataset

    #if specify a number of samples
    if k: 
        sample_indices = random.sample(range(len(test_dataset)), k)
    else:
        sample_indices = range(len(test_dataset))
    
    os.makedirs("outputs", exist_ok=True)
    

    generation_results = []
    
    for idx in sample_indices:
        example = test_dataset[idx]
        input_text = example["query"]
        
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
        

        input_length = inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)          
        print(f"\nGENERATED OUTPUT:\n{generated_text}")
        

        ground_truth = example['completion']
        print(f"\nGROUND TRUTH:\n{ground_truth}")
        

        generation_results.append({
            "prompt": input_text,
            "generation": generated_text,
            "ground_truth": ground_truth
        })
    
    #Save results to JSON
    if save:
        with open("outputs/countdown_generations.json", "w") as f:
            json.dump(generation_results, f, indent=2)
    else:
        return generation_results
    
    #print(f"\nGeneration results saved to outputs/generations.json")

if __name__ == "__main__":
  
    model_path = "training/final_model_v2" ##saif you'd prolly need to change this to finetuned model path
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please provide a valid path.")
        print("Using default model for demonstration purposes.")
        model_path = "Qwen/Qwen2.5-0.5B"  
    
    generate_samples(model_path, dataset="default") 