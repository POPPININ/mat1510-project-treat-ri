import yaml
from train.train import train_model, preprocess_data
from data.utils import split_data
import torch
import matplotlib.pyplot as plt


def plot_by_trs_weight(train_curves_by_trs, val_curves_by_trs, labels, trs_weights, save_path_train, save_path_val, optimizer_name, scheduler_name):
    """
    Plot and save the training and validation loss curves separately for each TRS weight.

    Args:
        train_curves_by_trs (dict): Dictionary of training curves for each TRS weight.
        val_curves_by_trs (dict): Dictionary of validation curves for each TRS weight.
        labels (list): Labels for each experiment.
        trs_weights (list): List of TRS weights for each experiment.
        save_path_train (str): Path template to save the training loss plots.
        save_path_val (str): Path template to save the validation loss plots.
        optimizer_name (str): Name of the optimizer used in training.
        scheduler_name (str): Name of the scheduler used in training.
    """
    markers = ['o', 's', '^', 'd', 'P']
    colors = ['r', 'g', 'b', 'm', 'c']
    linestyles = ['-', '--', '-.', ':', '-']

    for trs_weight in trs_weights:
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            train_curve = train_curves_by_trs[trs_weight][i]
            plt.plot(train_curve, label=f"Train Loss ({label})", 
                     marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curves (TRS Weight={trs_weight}, Optimizer={optimizer_name}, Scheduler={scheduler_name})")
        plt.legend()
        plt.grid(True)
        save_path = save_path_train.replace("trs_weight", f"trs_weight_{trs_weight}").replace("optimizer", optimizer_name).replace("scheduler", scheduler_name)
        plt.savefig(save_path)
        plt.show()
        print(f"Training loss plot for TRS Weight {trs_weight} saved to {save_path}")

        # Plot validation curves
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            val_curve = val_curves_by_trs[trs_weight][i]
            plt.plot(val_curve, label=f"Val Loss ({label})", 
                     marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Validation Loss Curves (TRS Weight={trs_weight}, Optimizer={optimizer_name}, Scheduler={scheduler_name})")
        plt.legend()
        plt.grid(True)
        save_path = save_path_val.replace("trs_weight", f"trs_weight_{trs_weight}").replace("optimizer", optimizer_name).replace("scheduler", scheduler_name)
        plt.savefig(save_path)
        plt.show()
        print(f"Validation loss plot for TRS Weight {trs_weight} saved to {save_path}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    optimizer_name = config["optimizer"]["name"]
    scheduler_name = config["scheduler"]["name"]

    data = torch.load(config["data"]["path"])
    timesteps = config["training"]["timesteps"]
    processed_data = preprocess_data(data, timesteps)

    train_data, val_data = split_data(processed_data, train_ratio=0.8, random_seed=config["random_seed"])

    # Experiment with different RI weights
    trs_weights = [0.5]
    ri_weights = [0.0, 0.5, 0.75, 1.0, 2.0]
    labels = [f"RotF={ri}" for ri in ri_weights]

    train_curves_by_trs = {trs: [] for trs in trs_weights}
    val_curves_by_trs = {trs: [] for trs in trs_weights}

    for trs_weight in trs_weights:
        print(f"Running Experiments for TRS Weight={trs_weight}...")
        for ri_weight in ri_weights:
            print(f"  Experiment (RI={ri_weight}, TRS={trs_weight})...")
            config_exp = config.copy()
            config_exp["training"]["ri_weight"] = ri_weight
            config_exp["training"]["trs_weight"] = trs_weight
            train_loss, val_loss = train_model(config_exp, train_data, val_data)
            train_curves_by_trs[trs_weight].append(train_loss)
            val_curves_by_trs[trs_weight].append(val_loss)

    plot_by_trs_weight(
        train_curves_by_trs,
        val_curves_by_trs,
        labels,
        trs_weights,
        save_path_train="training_loss_curves_trs_weight_optimizer_scheduler.png",
        save_path_val="validation_loss_curves_trs_weight_optimizer_scheduler.png",
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
    )
