import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from train.losses import TREATLoss
from train.reverse_operations import reverse_op
from train.train import preprocess_data  # Ensure this function is imported

def test_treat_loss():
    # Create dummy data
    T, batch_size, num_agents, dim = 10, 2, 3, 4
    z = torch.randn(T, batch_size, num_agents, dim)  # Forward trajectory
    z_pred = z + 0.1 * torch.randn_like(z)          # Noisy prediction
    z_rev_gt = torch.flip(z, dims=[0])              # Reverse ground truth
    z_rev_pred = torch.flip(z_pred, dims=[0])       # Reverse prediction

    # Initialize loss function
    loss_fn = TREATLoss(reverse_op=reverse_op)

    # Compute losses
    l_pred = loss_fn.prediction_loss(z, z_pred)
    l_rev_gt = loss_fn.time_reversal_loss_gt_rev(z_rev_gt, z_rev_pred)
    l_rev2 = loss_fn.time_reversal_loss_rev2(z, z_rev_pred)

    # Verify loss computation
    assert l_pred.item() > 0
    assert l_rev_gt.item() > 0
    assert l_rev2.item() > 0
    print("Loss tests passed!")


def test_preprocess_data():
    # Create dummy data
    num_samples, timesteps, num_agents, dim = 10, 100, 5, 4
    data = torch.randn(num_samples, timesteps, num_agents, dim)

    # Preprocess data
    processed_data = preprocess_data(data, timesteps)

    # Verify shapes
    assert processed_data["initial"].shape == (num_samples, num_agents, dim)
    assert processed_data["forward"].shape == (timesteps, num_samples, num_agents, dim)
    assert processed_data["reverse"].shape == (timesteps, num_samples, num_agents, dim)
    print("Preprocess data test passed!")


if __name__ == "__main__":
    test_treat_loss()
    test_preprocess_data()
