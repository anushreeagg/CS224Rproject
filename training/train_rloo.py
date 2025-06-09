import sys
import os
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from peft import PeftModel
from eval.generate_samples import generate_samples
from verl.utils.reward_score.countdown import compute_score
from outputs.convert_to_jsonl import extract_numbers_from_prompt

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

def compute_logprob(model, tokenizer, prompt, completion):
    """
    Compute log prob of completion given prompt.
    """
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs.input_ids
    labels = input_ids.clone()

    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

    token_log_probs = log_probs[:, :-1].gather(2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = labels[:, 1:] != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    return token_log_probs[mask].sum()

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

def train_rloo(sft_model_path, data, max_new_tokens=1024):
    """
    Training script to implement RLOO. 
    """
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto")
       
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    model.to(device)
    model.train()

    num_epochs = 3
    k = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        total_loss = 0.0 

        for row in tqdm(data):
            prompt = row["query"]
            completions = []

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            for _ in range(k):
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                completions.append(completion)

            rewards = []
            for completion in completions:
                ground_truth = {
                    "target":row["target"],
                    "numbers":row["nums"]
                }
                reward = compute_score(completion, ground_truth)
                rewards.append(reward)


            optimizer.zero_grad()
            batch_loss = 0.0

            for i in range(k):
                baseline = (sum(rewards) - rewards[i]) / (k - 1)
                advantage = rewards[i] - baseline

                log_prob = compute_logprob(model, tokenizer, prompt, completions[i])

                loss = -advantage * log_prob
                batch_loss += loss
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(data)
        logger.info(f"End of epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    model.save_pretrained("./rloo_model")
    tokenizer.save_pretrained("./rloo_model")




if __name__ == "__main__":
    # Load the data and the final sft model, then run the RLOO training script 
    data = load_dataset("https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    formatted_data = format_countdown_prompt(data)
    train_rloo("./training/final_model_v11", formatted_data)

