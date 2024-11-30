import yaml
import torch
import os
import matplotlib.pyplot as plt
from train.train import train_model, preprocess_data
from data.utils import split_data


def plot_optimizer_comparison(
    train_curves, val_curves, optimizers, ri_value, save_path_train, save_path_val
):
    """
    Plot and save training and validation loss curves for different optimizers.

    Args:
        train_curves (dict): Dictionary of training loss curves by optimizer.
        val_curves (dict): Dictionary of validation loss curves by optimizer.
        optimizers (list): List of optimizer names.
        ri_value (float): Rotational invariance weight.
        save_path_train (str): Path to save the training loss plot.
        save_path_val (str): Path to save the validation loss plot.
    """
    markers = ['o', 's', '^', 'd', '*']  # Markers for different optimizers
    colors = ['r', 'g', 'b', 'm', 'c']  # Colors for different optimizers
    linestyles = ['-', '--', '-.', ':', '-']  # Line styles for visual variety

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    for i, optimizer in enumerate(optimizers):
        plt.plot(
            train_curves[optimizer],
            label=f"{optimizer} (RI={ri_value})",
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Comparison (RI={ri_value})")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path_train), exist_ok=True)
    plt.savefig(save_path_train)
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    for i, optimizer in enumerate(optimizers):
        plt.plot(
            val_curves[optimizer],
            label=f"{optimizer} (RI={ri_value})",
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Comparison (RI={ri_value})")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path_val)
    plt.show()


if __name__ == "__main__":
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    data = torch.load(config["data"]["path"])
    timesteps = config["training"]["timesteps"]
    processed_data = preprocess_data(data, timesteps)

    # Split data with a fixed seed
    train_data, val_data = split_data(processed_data, train_ratio=0.8, random_seed=config["random_seed"])

    # Load TRS weight and Rotational Forward weight from config
    trs_weight = config["training"]["trs_weight"]
    rot_forward_weight = config["training"]["rot_forward_weight"]

    # Experiment parameters
    ri_values = [0.5, 0.75, 1.0, 2.0]
    optimizers = ["MomentumSGD", "AdamW"]

    # Store results
    for ri_value in ri_values:
        train_curves = {}
        val_curves = {}

        for optimizer in optimizers:
            print(f"Running Experiment: RI={ri_value}, TRS={trs_weight}, Optimizer={optimizer}...")
            config_exp = config.copy()
            config_exp["training"]["trs_weight"] = trs_weight
            config_exp["training"]["ri_weight"] = ri_value
            config_exp["training"]["rot_forward_weight"] = rot_forward_weight
            config_exp["optimizer"]["name"] = optimizer
            train_loss, val_loss = train_model(config_exp, train_data, val_data)
            train_curves[optimizer] = train_loss
            val_curves[optimizer] = val_loss

        # Plot results for this RI value
        plot_optimizer_comparison(
            train_curves,
            val_curves,
            optimizers,
            ri_value,
            save_path_train=f"comparison/training_loss_comparison_ri_{ri_value}.png",
            save_path_val=f"comparison/validation_loss_comparison_ri_{ri_value}.png",
        )
