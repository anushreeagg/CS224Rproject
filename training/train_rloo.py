import sys
import os
import re
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from peft import PeftModel
from verl.utils.reward_score.countdown import compute_score
from outputs.convert_to_jsonl import extract_numbers_from_prompt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_numbers_and_target(prompt_text):
    """
    Extracts numbers + the target value
    """
    numbers = extract_numbers_from_prompt(prompt_text)
    target = int(re.search(r'equals (\d+)', prompt_text).group(1))
    return numbers, target

def compute_logprob(model, tokenizer, prompts, completions):
    """
    Compute log prob of completion given prompts in batches.
    """
    log_probs = []
    for prompt, completion in zip(prompts, completions):

        full_text = prompt + completion
        inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)
        input_ids = inputs.input_ids

        prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device)
        prompt_len = prompt_inputs.input_ids.shape[1]

        if prompt_len >= input_ids.shape[1]:
            print(f"Warning: Prompt length ({prompt_len}) >= input length ({input_ids.shape[1]})")
            # Create a tensor that requires grad
            log_probs.append(torch.zeros(1, device=device, dtype=model.dtype, requires_grad=True))
            continue

        # Having issues with gradients fns not existing so ensuring we're in train mode
        model.train()

        outputs = model(input_ids=input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits
        
        # Get completion tokens only
        completion_tokens = input_ids[:, prompt_len:]
        completion_logits = logits[:, prompt_len-1:-1, :] 
        
        # Ensure shapes match
        if completion_tokens.shape[1] != completion_logits.shape[1]:
            min_len = min(completion_tokens.shape[1], completion_logits.shape[1])
            completion_tokens = completion_tokens[:, :min_len]
            completion_logits = completion_logits[:, :min_len, :]
        
        if completion_tokens.shape[1] == 0:
            # Create a tensor that requires grad for empty completions
            log_probs.append(torch.zeros(1, device=device, dtype=model.dtype, requires_grad=True))
            continue
        
        # Compute log probabilities
        log_probs_tensor = F.log_softmax(completion_logits, dim=-1)
        
        # Get log prob of actual tokens
        token_log_probs = log_probs_tensor.gather(2, completion_tokens.unsqueeze(-1)).squeeze(-1)
        completion_log_prob = token_log_probs.sum()
        
        log_probs.append(completion_log_prob)
    
    return torch.stack(log_probs)

def generate_completions_batch(model, tokenizer, prompts, k=4, max_new_tokens=1024):
    """
    Generate k completions using batch generation
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Repeat each prompt k times
    input_ids = input_ids.repeat_interleave(k, dim=0)
    attention_mask = attention_mask.repeat_interleave(k, dim=0)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    completions = []
    for i in range(len(prompts)):
        group = []
        for j in range(k):
            idx = i * k + j
            completion = tokenizer.decode(outputs[idx][input_ids.shape[1] - 1:], skip_special_tokens=True).strip()
            group.append(completion)
        completions.append(group)
    return completions

def format_countdown_prompt(dataset):
    """
    Returns the properly formatted prompt in the style of the cognitive behaviors dataset.
    """
    data = []
    for ex in dataset:
        numbers = ex["nums"]
        target = ex["target"]
        query = (f"A conversation between User and Assistant. The user asks a question, "
                        f"and the Assistant solves it. The assistant first thinks about the "
                        f"reasoning process in the mind and then provides the user with the answer.\n"
                        f"User: Using the numbers {numbers}, create an equation that equals "
                        f"{target}. You can use basic arithmetic operations (+, -, *, /) and each "
                        f"number can only be used once. Show your work in <think> </think> tags. And "
                        f"return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
                        f"Assistant: Let me solve this step by step.")
        data.append({
            "query":query,
            "completion":"",
            "nums":numbers,
            "target":target
        })
    return Dataset.from_list(data)

def train_rloo(sft_model_path, data, max_new_tokens=1024, subset=False, batch_size=4):
    """
    Training script to implement RLOO. 
    """
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32, device_map="auto", batch_size=4)
       
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    model.to(device)
    model.train()

    # Having issues w params not having gradients 
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            param.requires_grad = True
            print(f"Enabled gradients for: {name}")

    if subset:
        subset_size = int(2750)
        train_subset = data.select(range(subset_size))
    else:
        train_subset = data

    dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "query": [item["query"] for item in batch],
            "nums": [item["nums"] for item in batch],
            "target": [item["target"] for item in batch],
        }
    )

    num_epochs = 1
    k = 2
    gradient_accumulation_steps = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs * len(dataloader))

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        total_loss = 0.0 
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            prompts = batch["query"]
            targets = batch["target"]
            nums_list = batch["nums"]

            completions_list = generate_completions_batch(model, tokenizer, prompts, k, max_new_tokens)

            rewards_list = []
            for i, completion in enumerate(completions):
                rewards = []
                for completion in completions:
                    ground_truth = {
                        "target":row["target"],
                        "numbers":row["nums"]
                    }
                    reward = compute_score(completion, ground_truth)
                    rewards.append(reward)
                rewards_list.append(torch.tensor(rewards, device=device, dtype=torch.float16))

            rewards_stack = torch.stack(rewards_list)

            if rewards_stack.std() > 1e-6:
                rewards_stack = (rewards_stack - rewards_stack.mean()) / (rewards_stack.std() + 1e-8)

            model.train()

            flat_prompts = [p for p in prompts for _ in range(k)]
            flat_completions = [c for comps in completions_list for c in comps]
            log_probs = compute_logprob(model, tokenizer, flat_prompts, flat_completions)
            log_probs = log_probs.view(len(prompts), k)

            batch_loss = torch.zeros(1, device=device, dtype=torch.float16, requires_grad=True)
            # RLOO 
            for i in range(len(prompts)):
                rewards_sum = rewards_stack[i].sum()
                baseline = (rewards_sum.unsqueeze(0) - rewards_stack[i]) / (k - 1)
                advantages = rewards_stack[i] - baseline
                
                loss_contrib = (-advantages * log_probs[i]).sum()
                batch_loss = batch_loss + loss_contrib
            
            batch_loss = batch_loss / (len(prompts) * k * gradient_accumulation_steps)
            batch_loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += batch_loss.item() * gradient_accumulation_steps
            if step % 50 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"End of epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    model.save_pretrained("./rloo_model")
    tokenizer.save_pretrained("./rloo_model")

if __name__ == "__main__":
    # Load the data and the final sft model, then run the RLOO training script 
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    formatted_data = format_countdown_prompt(data)
    train_rloo("./training/final_model_v11", formatted_data)

