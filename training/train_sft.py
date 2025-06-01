import sys
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import math
import shutil


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

def get_cos_sched(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Returns a cosine schedule for the learning rate with a linear warmup 
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def save_checkpoint(model, tokenizer, optimizer, scaler, scheduler, epoch, global_step, checkpoint_dir, training_state={}):
    """
    Save a checkpoint of the training state
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(checkpoint_dir)
    else:
        model._orig_mod.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    training_state.update({
            'epoch':epoch,
            'global_step':global_step,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scaler_state_dict':scaler.state_dict(),
            'scheduler_state_dict':scheduler.state_dict() if scheduler else None,
            'lora_config':getattr(model, "peft_config", None)
        })
    torch.save(training_state, f"{checkpoint_dir}/training_state.pt")
    logger.info(f"Checkpoint saved at {checkpoint_dir}")

    


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

    #training hyperparams
    learning_rate = 5e-5
    num_epochs = 10
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    gradient_accumuluation_steps = 4

    #total training steps from gradient accumulation
    steps_per_epoch = len(train_dataloader) // gradient_accumuluation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    
    #optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    #scheduler
    scheduler = get_cos_sched(optimizer, warmup_steps, total_steps)

    #gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    #number of steps before saving
    save_steps = 200

    #saving training and validation losses
    training_losses = []
    validation_losses = []
    best_eval_loss = float('inf')
    best_model_epoch = 0
    
    #saving directory for final model + checkpoints
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./final_model", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    
    #the training loop
    logger.info("Starting training...")
    global_step = 0
    accumulation_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
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
                loss = loss / gradient_accumuluation_steps
            
            #backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            #Update metrics
            epoch_loss += loss.item() * gradient_accumuluation_steps
            num_batches += 1
            accumulation_step += 1

            if accumulation_step % gradient_accumuluation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
                #logging
                global_step += 1
                if global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    logger.info(f"Step {global_step} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}")
                    training_losses.append((global_step, loss.item()))
                

                #saving checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = f"./checkpoints/step_{global_step}"
                    save_checkpoint(model, tokenizer, optimizer, scaler, scheduler,
                                    epoch, global_step, checkpoint_dir)
                
            current_loss = epoch_loss / num_batches if num_batches > 0 else 0
            progress_bar.set_postfix({"loss": loss.item()})
            
        

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} finished. Average loss: {avg_epoch_loss:.4f}")

        if len(eval_dataloader) > 0:
            model.eval()
            eval_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    
                    eval_loss += outputs.loss.item()
                    num_batches += 1

            avg_eval_loss = eval_loss / num_batches if num_batches > 0 else 0
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_model_epoch = epoch + 1
                logger.info(f"New best model found at epoch {best_model_epoch} with validation loss {best_eval_loss:.4f}")

                save_checkpoint(model, tokenizer, optimizer, scaler, scheduler, epoch,
                                global_step, "./checkpoints/best_model",
                                {'best_eval_loss':best_eval_loss}
                            )
                
            logger.info(f"Validation loss: {avg_eval_loss:.4f}")
            validation_losses.append((epoch + 1, avg_eval_loss))
        
    
    logger.info("Saving training and validation losses")
    with open("training_losses.json", "w") as f:
        json.dump(training_losses, f)
    
    with open("validation_losses.json", "w") as f:
        json.dump(validation_losses, f)


    logger.info("Saving final model...")
    #in case there are issues with saving the compiled model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained("./final_model")
    else:
        model._orig_mod.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

    #copy final model to best_model directory
    if os.path.exists("./checkpoints/best_model"):
        logger.info(f"Copying best model (epoch {best_model_epoch}, loss: {best_eval_loss}) to ./best_model")
        for file in os.listdir("./checkpoints/best_model"):
            if file != "training_state.pt":
                src = os.path.join("./checkpoints/best_model", file)
                dst = os.path.join("./best_model", file)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    
    best_model_info = {
        "best_epoch": best_model_epoch,
        "best_eval_loss": best_eval_loss,
    }
    with open("./best_model/model_info.json", "w") as f:
        json.dump(best_model_info, f)

    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train() 