import torch
from .spatial_temporal_gnn import SpatialTemporalGNN
from .ode_function import GNNODEFunc
from .decoder import MultiAgentDecoder
from torchdiffeq import odeint


class MultiAgentTREAT(torch.nn.Module):
    def __init__(self, num_agents, edge_dim=1, hidden_dim=64, latent_dim=16, node_dim=4):
        """
        Initialize the MultiAgentTREAT model.

        Args:
            num_agents (int): Number of agents in the system.
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Hidden dimension for encoder and ODE function.
            latent_dim (int): Latent dimension for ODE function output.
            node_dim (int): Input dimension for node features.
        """
        super(MultiAgentTREAT, self).__init__()
        self.encoder = SpatialTemporalGNN(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.ode_func = GNNODEFunc(latent_dim=latent_dim)
        self.decoder = MultiAgentDecoder(latent_dim=latent_dim, output_dim=4)
        self.ode_solver = odeint

    def forward(self, node_features, edge_indices, t_span, edge_features=None):
        # Handle missing edge features
        if edge_features is None:
            edge_features = torch.ones(edge_indices.size(0), 1).to(node_features.device)  # Default to all-ones

        # Encode initial state
        z0 = self.encoder(node_features, edge_features, edge_indices)

        # Solve the ODE for the trajectory
        z_trajectory = self.ode_solver(self.ode_func, z0, t_span)

        # Decode trajectory
        decoded_trajectory = torch.stack([self.decoder(z) for z in z_trajectory])
        return decoded_trajectory

