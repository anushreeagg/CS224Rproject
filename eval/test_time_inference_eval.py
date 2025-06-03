import sys
import os
import torch
import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.generate_samples import transform_countdown

def generate_multiple_samples(model_path, max_new_tokens=1500):
    """
    Generating multiple outputs, which we'll score and then use the best scoring result. 
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        device_map="auto"
    )

    #fine-tuned model weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    test_dataset = transform_countdown("data/countdown.json")


if __name__ == '__main__':
    model_path = "training/final_model_v11"
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please provide a valid path.")
        print("Using default model for demonstration purposes.")
        model_path = "Qwen/Qwen2.5-0.5B"  

    generate_multiple_samples(model_path)
    