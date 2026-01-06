"""
Task-specific output heads for multi-task forecasting.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class TaskHead(nn.Module):
    """
    Single task output head.
    
    Transforms shared representation into task-specific predictions.
    
    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output (forecast_horizon * num_outputs)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_zones, input_dim)
            
        Returns:
            Output tensor of shape (batch, num_zones, output_dim)
        """
        return self.network(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task output heads with shared components.
    
    Creates separate heads for each task while sharing some layers.
    
    Args:
        input_dim: Dimension of input features
        tasks: List of task names
        forecast_horizon: Number of forecast steps
        num_zones: Number of zones (for zonal predictions)
        hidden_dim: Hidden layer dimension
        shared_layers: Number of shared layers before task-specific heads
    """
    
    def __init__(
        self,
        input_dim: int,
        tasks: List[str],
        forecast_horizon: int = 24,
        num_zones: int = 11,
        hidden_dim: int = 128,
        shared_layers: int = 1
    ):
        super().__init__()
        
        self.tasks = tasks
        self.forecast_horizon = forecast_horizon
        self.num_zones = num_zones
        
        # Shared layers
        shared = []
        shared.append(nn.Linear(input_dim, hidden_dim))
        shared.append(nn.ReLU())
        
        for _ in range(shared_layers - 1):
            shared.append(nn.Linear(hidden_dim, hidden_dim))
            shared.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*shared)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task in tasks:
            # Determine output dimension based on task type
            if task in ['load', 'lbmp']:
                # Zonal predictions
                output_dim = forecast_horizon
            elif task in ['emissions', 'renewable']:
                # System-wide predictions
                output_dim = forecast_horizon
            else:
                output_dim = forecast_horizon
            
            self.task_heads[task] = TaskHead(
                input_dim=hidden_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim // 2
            )
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task heads.
        
        Args:
            x: Input tensor of shape (batch, num_zones, input_dim)
            tasks: Optional subset of tasks to compute
            
        Returns:
            Dictionary of predictions for each task
        """
        tasks = tasks or self.tasks
        
        # Apply shared layers
        shared_out = self.shared_layers(x)  # (batch, num_zones, hidden_dim)
        
        predictions = {}
        
        for task in tasks:
            if task in self.task_heads:
                if task in ['load', 'lbmp']:
                    # Zonal prediction
                    pred = self.task_heads[task](shared_out)
                    # (batch, num_zones, forecast_horizon)
                    predictions[task] = pred.permute(0, 2, 1)
                    # (batch, forecast_horizon, num_zones)
                else:
                    # System-wide prediction (aggregate zones)
                    zone_avg = shared_out.mean(dim=1)  # (batch, hidden_dim)
                    pred = self.task_heads[task](zone_avg.unsqueeze(1))
                    predictions[task] = pred.squeeze(1)
                    # (batch, forecast_horizon)
        
        return predictions


class ProbabilisticTaskHead(nn.Module):
    """
    Probabilistic task head that outputs distribution parameters.
    
    Outputs mean and variance for Gaussian predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_samples: Whether to return Monte Carlo samples
            num_samples: Number of samples if return_samples is True
            
        Returns:
            Dictionary with 'mean', 'std', and optionally 'samples'
        """
        hidden = self.shared(x)
        
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)
        std = torch.exp(0.5 * logvar)
        
        output = {
            'mean': mean,
            'std': std,
            'logvar': logvar
        }
        
        if return_samples:
            # Reparameterization trick
            eps = torch.randn(num_samples, *mean.shape, device=mean.device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps
            output['samples'] = samples
        
        return output
