# CS224R Project - Supervised Fine-Tuning on Countdown Dataset

This repository contains code for supervised fine-tuning (SFT) of the Qwen2.5-0.5B model on the MMLU Countdown dataset.

## Project Structure

```
.
├── data/
│   ├── dataset.py           # CountdownDataset implementation
│   └── inspect_dataset.py   # Script to analyze dataset statistics
├── model/
│   └── model.py             # Model loading and LoRA configuration
├── training/
│   └── train_sft.py         # SFT training script
└── eval/
    └── generate_samples.py  # Evaluation script for sample generation
```

## Setup

1. Install the required dependencies:

```bash
pip install torch transformers datasets peft
```

2. Inspect the dataset:

```bash
python data/inspect_dataset.py
```

3. Train the model:

```bash
python training/train_sft.py
```

4. Generate samples using the fine-tuned model:

```bash
python eval/generate_samples.py
```

## Dataset

The project uses the "cais/mmlu_countdown" dataset from HuggingFace, which is designed for supervised fine-tuning.

## Model

The base model is Qwen2.5-0.5B ("Qwen/Qwen1.5-0.5B"), fine-tuned using Low-Rank Adaptation (LoRA) with the following configuration:
- r=8
- alpha=32
- dropout=0.1
- Target modules: ["c_attn", "q_proj", "v_proj"]

## Implementation Details

### CountdownDataset

The `CountdownDataset` class handles:
- Loading data from the HuggingFace dataset
- Tokenizing inputs and targets
- Truncating to specified maximum lengths (256 tokens for input, 1024 for output)
- Padding sequences to max length
- Masking padded tokens in labels with -100 (ignored in loss calculation)

### Training

The training script uses the Hugging Face Trainer API to fine-tune the model with the following features:
- LoRA for parameter-efficient fine-tuning
- Mixed precision training
- Checkpointing and evaluation during training

### Evaluation

The evaluation script generates responses to prompts from the test set and compares them with ground truth responses.

## License

[Add license information here] 