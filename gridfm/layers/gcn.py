"""
Zonal Graph Convolutional Network for modeling spatial dependencies
between power grid zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ZonalGCN(nn.Module):
    """
    Graph Convolutional Network for encoding zonal topology.
    
    This module captures spatial dependencies between different zones
    in the power grid using message passing on the adjacency graph.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden representations
        output_dim: Dimension of output features
        num_layers: Number of GCN layers
        num_zones: Number of zones in the grid
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_zones: int = 11,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_zones = num_zones
        
        # GCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(GCNLayer(hidden_dim, output_dim))
        
        # Attention mechanism for interpretability
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Zonal GCN.
        
        Args:
            x: Node features of shape (batch, num_zones, input_dim)
            adjacency: Adjacency matrix of shape (num_zones, num_zones)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Updated node features of shape (batch, num_zones, output_dim)
            attention: Optional attention weights
        """
        # Normalize adjacency matrix
        adj_norm = self._normalize_adjacency(adjacency)
        
        # Apply GCN layers
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_norm)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        # Apply attention for interpretability
        h_attn, attn_weights = self.attention(h, h, h)
        
        # Residual connection and normalization
        output = self.norm(h + h_attn)
        
        if return_attention:
            return output, attn_weights
        return output, None
    
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix using symmetric normalization.
        
        A_norm = D^(-1/2) * A * D^(-1/2)
        """
        # Add self-loops
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Symmetric normalization
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return adj_norm


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.
    
    Implements: H' = σ(A_norm * H * W)
    """
    
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
            
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GCN layer.
        
        Args:
            x: Node features of shape (batch, num_nodes, input_dim)
            adj_norm: Normalized adjacency matrix
            
        Returns:
            Updated node features of shape (batch, num_nodes, output_dim)
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        
        # Graph convolution (message passing)
        output = torch.matmul(adj_norm, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output


class AdaptiveGCN(nn.Module):
    """
    Adaptive GCN that learns the adjacency matrix from data.
    
    This is useful when the true connectivity between zones
    may differ from the physical topology.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_zones: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.num_zones = num_zones
        
        # Learnable node embeddings for adaptive adjacency
        self.node_embeddings = nn.Parameter(
            torch.randn(num_zones, hidden_dim) * 0.1
        )
        
        # Standard GCN
        self.gcn = ZonalGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_zones=num_zones
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        x: torch.Tensor,
        physical_adjacency: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with adaptive adjacency learning.
        
        Args:
            x: Node features
            physical_adjacency: Optional physical topology constraint
            return_attention: Whether to return attention
            
        Returns:
            output: Updated features
            attention: Optional attention weights
        """
        # Compute adaptive adjacency from node embeddings
        adaptive_adj = self._compute_adaptive_adjacency()
        
        # Combine with physical adjacency if provided
        if physical_adjacency is not None:
            # Use physical adjacency as mask/prior
            adjacency = adaptive_adj * physical_adjacency
        else:
            adjacency = adaptive_adj
        
        return self.gcn(x, adjacency, return_attention)
    
    def _compute_adaptive_adjacency(self) -> torch.Tensor:
        """Compute adaptive adjacency from learned embeddings."""
        # Compute pairwise similarities
        similarity = torch.matmul(
            self.node_embeddings,
            self.node_embeddings.T
        )
        
        # Apply temperature-scaled softmax
        adjacency = F.softmax(similarity / self.temperature, dim=-1)
        
        # Make symmetric
        adjacency = (adjacency + adjacency.T) / 2
        
        return adjacency
