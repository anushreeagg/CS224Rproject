from torch.utils.data import Dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Union


class CountdownDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_input_length=256, max_output_length=1024):
        """
        Initialize the CountdownDataset.
        
        Args:
            tokenizer: HuggingFace tokenizer
            split: Dataset split (train, validation, test)
            max_input_length: Maximum token length for inputs
            max_output_length: Maximum token length for outputs
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load the MMLU Countdown dataset
        self.dataset = load_dataset("cais/mmlu_countdown", split=split)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        # Tokenize input
        input_tokens = self.tokenizer(
            example["input"],
            truncation=True,
            max_length=self.max_input_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target/output
        target_tokens = self.tokenizer(
            example["target"],
            truncation=True,
            max_length=self.max_output_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create input_ids by concatenating input and target
        input_ids = torch.cat([
            input_tokens.input_ids.squeeze(0),
            target_tokens.input_ids.squeeze(0)
        ])
        
        # Create attention mask
        attention_mask = torch.cat([
            input_tokens.attention_mask.squeeze(0),
            target_tokens.attention_mask.squeeze(0)
        ])
        
        # Create labels: -100 for input tokens, actual ids for target tokens
        labels = torch.cat([
            torch.full_like(input_tokens.input_ids.squeeze(0), -100),
            target_tokens.input_ids.squeeze(0)
        ])
        
        # Replace padding tokens in labels with -100
        labels = torch.where(attention_mask == 1, labels, torch.tensor(-100))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        } 