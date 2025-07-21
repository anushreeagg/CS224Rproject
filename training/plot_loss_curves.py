import json
import matplotlib.pyplot as plt
import os

def plot_loss_curves(training_loss_file, validation_loss_file, train_path, val_path):
    """
    Plots training and validation loss curves from json files. 
    """
    if not os.path.exists(training_loss_file) or not os.path.exists(validation_loss_file):
        print("loss files not found.")
        return
    
    with open(training_loss_file) as f:
        training_losses = json.load(f)
    
    with open(validation_loss_file) as f:
        validation_losses = json.load(f)

    if not training_losses or not validation_losses:
        print("No data found.")
        return
    
    train_steps, train_loss_values = zip(*training_losses)
    val_steps, val_loss_values = zip(*validation_losses)

    # plotting training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss_values, label="Training Loss", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout
    plt.savefig(train_path)
    print(f"Saved training loss plot to {train_path}")
    plt.close()

    # plotting validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(val_steps, val_loss_values, label="Validation Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.tight_layout
    plt.savefig(val_path)
    print(f"Saved training loss plot to {val_path}")
    plt.close()

if __name__ == "__main__":
    plot_loss_curves("training_losses.json", "validation_losses.json", "training_curve.png", "validation_curve.png")


