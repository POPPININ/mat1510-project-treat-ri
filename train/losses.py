from torch import nn
import torch

class TREATLoss(nn.Module):
    def __init__(self, reverse_op):
        """
        Initialize TREATLoss with a reverse operation.

        Args:
            reverse_op (callable): A function that performs time reversal on trajectories.
        """
        super(TREATLoss, self).__init__()
        self.reverse_op = reverse_op

    def prediction_loss(self, z, z_pred):
        """
        Compute the prediction loss (forward trajectory alignment).

        Args:
            z (torch.Tensor): Ground truth forward trajectories.
            z_pred (torch.Tensor): Predicted forward trajectories.

        Returns:
            torch.Tensor: Prediction loss.
        """
        return torch.mean(torch.sum((z - z_pred) ** 2, dim=-1))

    def time_reversal_loss_gt_rev(self, z_rev_gt, z_rev_pred):
        """
        Compute the time-reversal loss with ground truth reverse trajectories.

        Args:
            z_rev_gt (torch.Tensor): Ground truth reverse trajectories.
            z_rev_pred (torch.Tensor): Predicted reverse trajectories.

        Returns:
            torch.Tensor: Time-reversal loss.
        """
        return torch.mean(torch.sum((z_rev_gt - z_rev_pred) ** 2, dim=-1))

    def time_reversal_loss_rev2(self, z, z_rev_pred):
        """
        Compute the time-reversal loss using reverse operation on predictions.

        Args:
            z (torch.Tensor): Ground truth forward trajectories.
            z_rev_pred (torch.Tensor): Predicted reverse trajectories.

        Returns:
            torch.Tensor: Time-reversal loss.
        """
        z_rev_pred_flipped = self.reverse_op(z_rev_pred)
        return torch.mean(torch.sum((z - z_rev_pred_flipped) ** 2, dim=-1))

    def generate_random_rotation_matrix(self, feature_dim, device, dtype):
        """
        Generate a random rotation matrix for 2D or 3D.

        Args:
            feature_dim (int): Dimensionality of the feature space.
            device: Torch device.
            dtype: Torch data type.

        Returns:
            torch.Tensor: Rotation matrix of shape (feature_dim, feature_dim).
        """
        if feature_dim == 2:
            theta = torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), -torch.sin(theta)],
                 [torch.sin(theta), torch.cos(theta)]],
                device=device, dtype=dtype
            ).squeeze(0)
        elif feature_dim == 3:
            random_vector = torch.randn(3, device=device, dtype=dtype)
            axis = random_vector / torch.norm(random_vector)
            angle = torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
            K = torch.tensor([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]], device=device, dtype=dtype)
            rotation_matrix = torch.eye(3, device=device, dtype=dtype) + \
                              torch.sin(angle) * K + \
                              (1 - torch.cos(angle)) * (K @ K)
        else:
            raise ValueError("Only 2D or 3D rotations are supported.")
        return rotation_matrix

    def rotational_forward_loss(self, model, x, edge_indices, rotation_matrix, timesteps):
        """
        Compute the rotational forward loss by applying a random rotation to the input
        and minimizing the difference between the original and rotated predictions.

        Args:
            model (torch.nn.Module): The model to evaluate.
            x (torch.Tensor): Initial positions and features (batch_size, num_agents, feature_dim).
            edge_indices (torch.Tensor): Edge indices for the graph.
            rotation_matrix (torch.Tensor): Rotation matrix for applying SO(2) or SO(3) transformations.
            timesteps (int): Number of timesteps to predict.

        Returns:
            torch.Tensor: Rotational forward loss.
        """
        feature_dim = rotation_matrix.size(0)  # Infer feature dimension from rotation matrix
        x_positions = x[..., :feature_dim]  # Extract only the positional dimensions
        x_non_positions = x[..., feature_dim:]  # Separate non-positional dimensions (if any)

        # Apply rotation to positional dimensions
        x_rot = torch.einsum("ij,bnj->bni", rotation_matrix, x_positions)

        # Combine rotated positions with non-positional dimensions
        x_rot_full = torch.cat([x_rot, x_non_positions], dim=-1)

        # Generate predictions for original and rotated inputs
        z_pred = model(x, edge_indices, torch.linspace(0, 1, timesteps).to(x.device))
        z_rot_pred = model(x_rot_full, edge_indices, torch.linspace(0, 1, timesteps).to(x.device))

        # Compute rotational forward loss (MSE between predictions)
        return torch.mean((z_pred - z_rot_pred) ** 2)

    def forward(
        self, z, z_pred, z_rev_gt=None, z_rev_pred=None, model=None, x=None, edge_indices=None,
        mode="gt-rev", trs_weight=1.0, include_rot_forward_loss=False, rot_forward_weight=0.0,
        feature_dim=2, timesteps=10
    ):
        """
        Compute the overall TREAT loss, including prediction, time-reversal, and rotational forward losses.

        Args:
            model (nn.Module): The model for forward trajectory predictions (required for rotational forward loss).
            x (torch.Tensor): Input initial states (required for rotational forward loss).
            edge_indices (torch.Tensor): Edge indices for the graph (required for rotational forward loss).
            include_rot_forward_loss (bool): Whether to include rotational forward loss.
            rot_forward_weight (float): Weight for rotational forward loss.

        Returns:
            torch.Tensor: Total loss.
        """
        # Prediction loss
        l_pred = self.prediction_loss(z, z_pred)

        # Time-reversal loss
        if mode == "gt-rev":
            assert z_rev_gt is not None, "Ground truth reverse trajectory required for gt-rev mode"
            l_reverse = self.time_reversal_loss_gt_rev(z_rev_gt, z_rev_pred)
        elif mode == "rev2":
            l_reverse = self.time_reversal_loss_rev2(z, z_rev_pred)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Rotational forward loss (optional)
        if include_rot_forward_loss:
            rotation_matrix = self.generate_random_rotation_matrix(feature_dim, x.device, x.dtype)
            l_rot_forward = self.rotational_forward_loss(model, x, edge_indices, rotation_matrix, timesteps)
        else:
            l_rot_forward = 0.0

        # Total loss
        return l_pred + trs_weight * l_reverse + rot_forward_weight * l_rot_forward
