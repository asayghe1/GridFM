"""
Physics-informed power balance constraint module.

Implements the fundamental power balance equation:
    ΣP_gen + P_storage = P_load + ΣP_flow + P_loss

This constraint ensures predictions respect conservation of energy.
"""

import torch
import torch.nn as nn
from typing import Optional


class PowerBalanceLoss(nn.Module):
    """
    Power balance constraint loss for physics-informed learning.
    
    The power balance equation states that at any time, the total
    generation plus storage discharge must equal load plus transmission
    losses plus storage charging.
    
    Loss = ||ΣP_gen + P_storage - P_load - ΣP_flow - P_loss||²
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
        loss_scale: float = 1.0
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.normalize = normalize
        self.loss_scale = loss_scale
        
    def forward(
        self,
        load_pred: torch.Tensor,
        generation_pred: Optional[torch.Tensor] = None,
        phase_angles: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        storage_pred: Optional[torch.Tensor] = None,
        line_reactance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute power balance constraint loss.
        
        Args:
            load_pred: Predicted load (batch, horizon, num_zones)
            generation_pred: Predicted generation (batch, horizon, num_zones)
            phase_angles: Predicted voltage phase angles (batch, horizon, num_zones)
            adjacency: Adjacency matrix (num_zones, num_zones)
            storage_pred: Predicted storage power (positive=discharge)
            line_reactance: Line reactance values for power flow
            
        Returns:
            Power balance loss value
        """
        # Total load
        total_load = load_pred.sum(dim=-1)  # (batch, horizon)
        
        # Total generation (if provided)
        total_gen = torch.zeros_like(total_load)
        if generation_pred is not None:
            total_gen = generation_pred.sum(dim=-1)
        
        # Storage contribution (if provided)
        storage_power = torch.zeros_like(total_load)
        if storage_pred is not None:
            storage_power = storage_pred.sum(dim=-1)
        
        # Compute power flows if phase angles available
        total_flow = torch.zeros_like(total_load)
        if phase_angles is not None and adjacency is not None:
            total_flow = self._compute_power_flows(
                phase_angles, adjacency, line_reactance
            )
        
        # Power balance: gen + storage = load + flow + losses
        # Losses approximated as percentage of load
        estimated_losses = 0.03 * total_load  # ~3% losses
        
        imbalance = total_gen + storage_power - total_load - total_flow - estimated_losses
        
        # Compute loss
        if self.normalize:
            # Normalize by load magnitude
            imbalance = imbalance / (total_load.abs() + 1e-8)
        
        if self.loss_type == "mse":
            loss = (imbalance ** 2).mean()
        elif self.loss_type == "mae":
            loss = imbalance.abs().mean()
        elif self.loss_type == "huber":
            loss = nn.functional.smooth_l1_loss(
                imbalance, 
                torch.zeros_like(imbalance)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return self.loss_scale * loss
    
    def _compute_power_flows(
        self,
        phase_angles: torch.Tensor,
        adjacency: torch.Tensor,
        line_reactance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute power flows using DC power flow approximation.
        
        P_ij = (θ_i - θ_j) / X_ij
        
        Args:
            phase_angles: Voltage phase angles (batch, horizon, num_zones)
            adjacency: Adjacency matrix indicating connections
            line_reactance: Line reactance values (default: 1.0 p.u.)
            
        Returns:
            Net power flow at each bus (batch, horizon)
        """
        batch_size, horizon, num_zones = phase_angles.shape
        
        # Default reactance
        if line_reactance is None:
            line_reactance = torch.ones_like(adjacency)
        
        # Compute angle differences for each edge
        # θ_i - θ_j for all i, j pairs
        angle_diff = phase_angles.unsqueeze(-1) - phase_angles.unsqueeze(-2)
        # (batch, horizon, num_zones, num_zones)
        
        # Power flow on each line: P_ij = (θ_i - θ_j) / X_ij
        # Only where connection exists (adjacency > 0)
        susceptance = adjacency / (line_reactance + 1e-8)
        power_flow = angle_diff * susceptance.unsqueeze(0).unsqueeze(0)
        
        # Net injection at each bus: sum of outgoing flows
        net_injection = power_flow.sum(dim=-1)  # (batch, horizon, num_zones)
        
        # Total net flow (should be close to zero for balanced system)
        total_flow = net_injection.sum(dim=-1)  # (batch, horizon)
        
        return total_flow


class ZonalPowerBalance(nn.Module):
    """
    Zonal power balance constraint for multi-zone grids.
    
    Ensures power balance at each zone individually:
    P_gen_i + P_in_i = P_load_i + P_out_i + P_loss_i
    """
    
    def __init__(self, num_zones: int):
        super().__init__()
        self.num_zones = num_zones
        
    def forward(
        self,
        load_pred: torch.Tensor,
        generation_pred: torch.Tensor,
        interchange_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute zonal power balance loss.
        
        Args:
            load_pred: Zone load predictions (batch, horizon, num_zones)
            generation_pred: Zone generation predictions
            interchange_pred: Inter-zone interchange predictions
            
        Returns:
            Zonal balance loss
        """
        # Zonal imbalance
        imbalance = generation_pred + interchange_pred - load_pred
        
        # Loss is sum of squared imbalances across all zones
        loss = (imbalance ** 2).mean()
        
        return loss
