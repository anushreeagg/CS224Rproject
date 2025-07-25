from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def load_model_and_tokenizer():
    """
    Load the Qwen2.5-0.5B model and tokenizer and apply LoRA for fine-tuning.
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    #loqd model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    #set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    #LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    #applyig LoRA
    model = get_peft_model(model, lora_config)
    
    
    model.print_trainable_parameters()
    
    return model, tokenizer, lora_config

if __name__ == "__main__":

    model, tokenizer, _ = load_model_and_tokenizer()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}") 