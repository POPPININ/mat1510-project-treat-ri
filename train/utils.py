import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses over epochs.

    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()
