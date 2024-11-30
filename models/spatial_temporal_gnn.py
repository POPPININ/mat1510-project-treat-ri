import torch
from torch import nn
import torch_scatter

class SpatialTemporalGNN(nn.Module):
    def __init__(self, node_dim=4, edge_dim=1, hidden_dim=64, latent_dim=16, temporal_dim=0):
        super(SpatialTemporalGNN, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)

    def forward(self, node_features, edge_features, edge_indices):
        """
        Forward pass for SpatialTemporalGNN.

        Args:
            node_features (torch.Tensor): Input node features (batch_size, num_agents, node_dim).
            edge_features (torch.Tensor): Input edge features (num_edges, edge_dim).
            edge_indices (torch.Tensor): Edge indices for graph (num_edges, 2).

        Returns:
            torch.Tensor: Latent node embeddings.
        """
        batch_size, num_agents, node_dim = node_features.size()
        
        # Flatten node_features for processing
        node_features = node_features.reshape(batch_size * num_agents, node_dim)

        # Compute initial node embeddings
        z_nodes = self.node_mlp(node_features)  # Shape: (batch_size * num_agents, latent_dim)

        # Compute edge embeddings
        z_edges = self.edge_mlp(edge_features)  # Shape: (num_edges, latent_dim)

        # Message passing: Aggregate edge features to update nodes
        source_nodes, target_nodes = edge_indices.t()  # Split edge indices
        
        # Ensure valid dim_size for scatter_add
        dim_size = batch_size * num_agents
        messages = torch.zeros_like(z_nodes)  # Initialize missing messages with zeros
        messages = torch_scatter.scatter_add(z_edges, target_nodes, dim=0, out=messages)

        # Combine messages with node embeddings
        z_nodes += messages

        # Self-attention
        z_nodes = z_nodes.view(batch_size, num_agents, -1).permute(1, 0, 2)  # (num_agents, batch_size, latent_dim)
        z_nodes, _ = self.self_attention(z_nodes, z_nodes, z_nodes)  # Self-attention on nodes
        z_nodes = z_nodes.permute(1, 0, 2)  # (batch_size, num_agents, latent_dim)

        return z_nodes
