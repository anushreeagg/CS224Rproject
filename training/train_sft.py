import sys
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CountdownDataset
from model.model import load_model_and_tokenizer

#model logging configuration:
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def train():
    """
    This function trainss the model using SFT with the Countdown dataset.
    Uses a custom PyTorch training loop with mixed precision.
    """
    #load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer()
    #move model to GPU (for saif's training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #use torch.compile for speedup
    try:
        model = torch.compile(model)
        logger.info("Successfully applied torch.compile for speedup")
    except Exception as e:
        logger.warning(f"Could not apply torch.compile:{e}")

    #create datasets + dataloaders
    train_dataset = CountdownDataset(tokenizer, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    #validation dataset
    eval_dataset = CountdownDataset(tokenizer, split="test")
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    #gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    #number of epochs
    num_epochs = 7

    #number of steps before saving
    save_steps = 200

    #saving training and validation losses
    training_losses = []
    validation_losses = []
    
    #saving directory for final model + checkpoints
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./final_model", exist_ok=True)
    
    #the training loop
    logger.info("Starting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            #move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            
            with torch.cuda.amp.autocast(): #forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            #backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            #Update metrics
            epoch_loss += loss.item()
            
            #logging
            global_step += 1
            if global_step % 10 == 0:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f}")
                training_losses.append((global_step, loss.item()))

            #saving checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = f"./checkpoints/step_{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                torch.save({
                    'epoch':epoch,
                    'global_step':global_step,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scaler_state_dict':scaler.state_dict(),
                    'lora_config':getattr(model, "peft_config", None)
                }, f"{checkpoint_dir}/training_state.pt")
            
            progress_bar.set_postfix({"loss": loss.item()})
        

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} finished. Average loss: {avg_epoch_loss:.4f}")
        

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
            validation_losses.append((epoch + 1, avg_eval_loss))
    
    logger.info("Saving training and validation losses")
    with open("training_losses.json", "w") as f:
        json.dump(training_losses, f)
    
    with open("validation_losses.json", "w") as f:
        json.dump(validation_losses, f)


    logger.info("Saving model...")
    #in case there are issues with saving the compiled model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained("./final_model")
    else:
        model._orig_mod.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train() 