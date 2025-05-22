from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def load_model_and_tokenizer():
    """
    Load the Qwen2.5-0.5B model and tokenizer and apply LoRA for fine-tuning.
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "c_attn",
            "q_proj",
            "v_proj"
        ],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer, lora_config

if __name__ == "__main__":
    # Test the model loading
    model, tokenizer, _ = load_model_and_tokenizer()
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}") 