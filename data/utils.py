import torch


def split_data(data, train_ratio=0.8, random_seed=None):
    """
    Split data into training and validation sets.

    Args:
        data (dict): Dictionary containing "initial", "forward", and "reverse".
        train_ratio (float): Ratio of the data to use for training.
        random_seed (int): Seed for reproducible random splits.

    Returns:
        tuple: (train_data, val_data) dictionaries containing the split data.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    num_samples = data["initial"].size(0)  # Number of samples
    train_size = int(train_ratio * num_samples)
    indices = torch.randperm(num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = {}
    val_data = {}

    for key, val in data.items():
        if key in ["forward", "reverse"]:
            # Apply indices to the second dimension
            train_data[key] = val[:, train_indices]
            val_data[key] = val[:, val_indices]
        else:
            # Apply indices to the first dimension
            train_data[key] = val[train_indices]
            val_data[key] = val[val_indices]

    return train_data, val_data

