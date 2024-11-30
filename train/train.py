import torch
from train.losses import TREATLoss
from train.reverse_operations import reverse_op
from models.treat_model import MultiAgentTREAT
from data.utils import split_data

def load_data(config):
    """
    Load data from the specified path in the configuration file.
    """
    print(f"Loading data from {config['data']['path']}...")
    data = torch.load(config["data"]["path"])
    print(f"Loaded data of shape: {data.shape}")
    return data

def preprocess_data(data, timesteps):
    """
    Preprocess the data into initial states, forward trajectories, and reverse trajectories.

    Args:
        data (torch.Tensor): Tensor of shape (num_samples, total_timesteps, num_agents, 4).
        timesteps (int): Number of timesteps for training.

    Returns:
        dict: Processed data with keys "initial", "forward", "reverse".
    """
    num_samples, total_timesteps, num_agents, feature_dim = data.size()
    assert total_timesteps >= timesteps, "Total timesteps must be greater than or equal to the required timesteps."

    initial = data[:, 0, :, :]  # Shape: (num_samples, num_agents, feature_dim)
    forward = data[:, :timesteps, :, :].permute(1, 0, 2, 3)  # Shape: (timesteps, num_samples, num_agents, feature_dim)
    reverse = data[:, -timesteps:, :, :].flip(dims=[1]).permute(1, 0, 2, 3)  # Shape: (timesteps, num_samples, num_agents, feature_dim)

    print(f"Training data shapes: Initial: {initial.shape}, Forward: {forward.shape}, Reverse: {reverse.shape}")
    return {"initial": initial, "forward": forward, "reverse": reverse}

def match_timesteps(data, target_timesteps):
    """
    Resample the given data to match the target number of timesteps.

    Args:
        data (torch.Tensor): Input data of shape (timesteps, batch_size, num_agents, feature_dim).
        target_timesteps (int): Target number of timesteps for resampling.

    Returns:
        torch.Tensor: Resampled data with target_timesteps.
    """
    timesteps, batch_size, num_agents, feature_dim = data.shape
    reshaped_data = data.permute(1, 2, 3, 0).reshape(-1, feature_dim, timesteps)
    resampled_data = torch.nn.functional.interpolate(
        reshaped_data, size=target_timesteps, mode="linear", align_corners=True
    )
    resampled_data = resampled_data.view(batch_size, num_agents, feature_dim, target_timesteps).permute(3, 0, 1, 2)
    return resampled_data

def load_optimizer(config, model_params):
    optimizer_name = config["optimizer"]["name"]
    params = config["optimizer"]["params"]
    print("Using Optimizer: %s" % optimizer_name)

    if optimizer_name == "SGD":
        return torch.optim.SGD(model_params, **params)
    elif optimizer_name == "MomentumSGD":
        return torch.optim.SGD(
            model_params,
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params.get("weight_decay", 0),
            nesterov=params.get("nesterov", False),
        )
    elif optimizer_name == "AdamW":
        adam_params = params.copy()
        adam_params.pop("momentum", None)
        adam_params.pop("nesterov", None)
        return torch.optim.AdamW(model_params, **adam_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def load_scheduler(config, optimizer):
    scheduler_name = config["scheduler"]["name"]
    params = config["scheduler"]["params"]

    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
    elif scheduler_name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["T_max"])
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params["factor"], patience=params["patience"])
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def train_model(config, train_data, val_data):
    device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Initialize model, optimizer, and scheduler
    model = MultiAgentTREAT(
        num_agents=train_data["initial"].size(1),
        hidden_dim=config["model"]["hidden_dim"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)

    optimizer = load_optimizer(config, model.parameters())
    scheduler = load_scheduler(config, optimizer)

    loss_fn = TREATLoss(reverse_op=reverse_op)
    edge_indices = torch.arange(0, train_data["initial"].size(1)).unsqueeze(1).repeat(1, 2).to(device)

    train_losses, val_losses = [], []

    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()

        # Move data to device
        train_data = {key: val.to(device) for key, val in train_data.items()}
        z = train_data["forward"]
        z_rev_gt = train_data["reverse"]

        # Forward and reverse pass
        z_pred = model(train_data["initial"], edge_indices, torch.linspace(0, 1, config["training"]["timesteps"]).to(device))
        z_pred_resampled = match_timesteps(z_pred, target_timesteps=z.size(0))

        z_rev_pred = model(train_data["reverse"][0], edge_indices, torch.linspace(0, 1, config["training"]["timesteps"]).to(device))
        z_rev_pred_resampled = match_timesteps(z_rev_pred, target_timesteps=z_rev_gt.size(0))

        # Loss calculation (use resampled predictions)
        loss = loss_fn(
            z, z_pred_resampled,  # Use resampled forward prediction
            z_rev_gt=z_rev_gt, z_rev_pred=z_rev_pred_resampled,  # Use resampled reverse prediction
            model=model, x=train_data["initial"],  # Pass initial trajectories
            edge_indices=edge_indices,
            mode=config["training"]["loss_mode"],
            trs_weight=config["training"].get("trs_weight", 1.0),
            include_rot_forward_loss=config["training"].get("include_rot_forward_loss", False),
            rot_forward_weight=config["training"].get("rot_forward_weight", 0.0),
            feature_dim=config["model"]["feature_dim"],
            timesteps=config["training"]["timesteps"]
        )
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_data = {key: val.to(device) for key, val in val_data.items()}
            z_val = val_data["forward"]
            z_val_rev_gt = val_data["reverse"]

            z_val_pred = model(val_data["initial"], edge_indices, torch.linspace(0, 1, config["training"]["timesteps"]).to(device))
            z_val_pred_resampled = match_timesteps(z_val_pred, target_timesteps=z_val.size(0))

            z_val_rev_pred = model(val_data["reverse"][0], edge_indices, torch.linspace(0, 1, config["training"]["timesteps"]).to(device))
            z_val_rev_pred_resampled = match_timesteps(z_val_rev_pred, target_timesteps=z_val_rev_gt.size(0))

            val_loss = loss_fn(
                z_val, z_val_pred_resampled,  # Use resampled forward validation
                z_rev_gt=z_val_rev_gt, z_rev_pred=z_val_rev_pred_resampled,  # Use resampled reverse validation
                model=model, x=val_data["initial"],  # Pass initial trajectories
                edge_indices=edge_indices,
                mode=config["training"]["loss_mode"],
                trs_weight=config["training"].get("trs_weight", 1.0),
                include_rot_forward_loss=config["training"].get("include_rot_forward_loss", False),
                rot_forward_weight=config["training"].get("rot_forward_weight", 0.0),
                feature_dim=config["model"]["feature_dim"],
                timesteps=config["training"]["timesteps"]
            )
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.6f}, Validation Loss: {val_losses[-1]:.6f}")

        # Step the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_losses[-1])  # Plateau scheduler needs validation loss
        else:
            scheduler.step()

    return train_losses, val_losses

