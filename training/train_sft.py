import sys
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.dataset import CountdownDataset
from model.model import load_model_and_tokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def train():
    """
    Train the model using SFT with the Countdown dataset.
    Uses a custom PyTorch training loop with mixed precision.
    """
    # Load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = CountdownDataset(tokenizer, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Optional validation dataset
    eval_dataset = CountdownDataset(tokenizer, split="validation")
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Set up gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Number of epochs
    num_epochs = 3
    
    # Create directories for saving
    os.makedirs("./final_model", exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Log progress
            global_step += 1
            if global_step % 10 == 0:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f}")
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} finished. Average loss: {avg_epoch_loss:.4f}")
        
        # Optional validation
        if len(eval_dataloader) > 0:
            model.eval()
            eval_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    eval_loss += outputs.loss.item()
            
            avg_eval_loss = eval_loss / len(eval_dataloader)
            logger.info(f"Validation loss: {avg_eval_loss:.4f}")
    
    # Save the final model
    logger.info("Saving model...")
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train() 