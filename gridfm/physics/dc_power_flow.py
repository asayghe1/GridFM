"""
DC Power Flow constraint module for physics-informed learning.

Implements the DC power flow approximation which relates power injections
to voltage phase angles through the network susceptance matrix.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DCPowerFlowConstraint(nn.Module):
    """
    DC Power Flow constraint for physics-informed predictions.
    
    The DC power flow equations:
        P = B * θ
        P_ij = (θ_i - θ_j) / X_ij
    
    where:
        P: Power injection vector
        B: Susceptance matrix
        θ: Voltage phase angle vector
        X_ij: Line reactance between buses i and j
    
    Args:
        num_zones: Number of zones/buses in the network
        base_mva: Base MVA for per-unit conversion (default: 100)
    """
    
    def __init__(
        self,
        num_zones: int,
        base_mva: float = 100.0
    ):
        super().__init__()
        
        self.num_zones = num_zones
        self.base_mva = base_mva
        
        # Learnable susceptance matrix (initialized from topology)
        self.susceptance = nn.Parameter(
            torch.zeros(num_zones, num_zones),
            requires_grad=True
        )
        
        # Reference bus (slack bus) - angles are relative to this
        self.reference_bus = 0
        
    def initialize_from_topology(
        self,
        adjacency: torch.Tensor,
        line_reactance: torch.Tensor
    ) -> None:
        """
        Initialize susceptance matrix from network topology.
        
        Args:
            adjacency: Binary adjacency matrix
            line_reactance: Line reactance values in per-unit
        """
        # Susceptance = 1 / reactance
        with torch.no_grad():
            B = torch.zeros_like(self.susceptance)
            
            for i in range(self.num_zones):
                for j in range(self.num_zones):
                    if adjacency[i, j] > 0 and i != j:
                        b_ij = 1.0 / (line_reactance[i, j] + 1e-8)
                        B[i, j] = -b_ij
                        B[i, i] += b_ij
            
            self.susceptance.data = B
            
    def forward(
        self,
        power_injection: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solve DC power flow to get phase angles from power injections.
        
        Args:
            power_injection: Net power injection at each bus (batch, horizon, num_zones)
            adjacency: Optional adjacency matrix to update susceptance
            
        Returns:
            Phase angles at each bus (batch, horizon, num_zones)
        """
        batch_size, horizon, num_zones = power_injection.shape
        
        # Get susceptance matrix (optionally masked by adjacency)
        B = self.susceptance
        if adjacency is not None:
            # Mask off-diagonal elements by adjacency
            mask = adjacency + torch.eye(num_zones, device=B.device)
            B = B * mask
        
        # Remove reference bus row/column for solvability
        B_reduced = self._remove_reference_bus(B)
        P_reduced = self._remove_reference_bus_vector(power_injection)
        
        # Solve: θ = B^(-1) * P
        # Use pseudo-inverse for numerical stability
        B_pinv = torch.linalg.pinv(B_reduced)
        
        # Reshape for batch processing
        P_flat = P_reduced.reshape(-1, num_zones - 1)  # (batch*horizon, num_zones-1)
        theta_flat = torch.matmul(P_flat, B_pinv.T)
        
        # Reshape back
        theta_reduced = theta_flat.reshape(batch_size, horizon, num_zones - 1)
        
        # Insert reference bus angle (0)
        theta = self._insert_reference_bus(theta_reduced)
        
        return theta
    
    def compute_loss(
        self,
        phase_angles: torch.Tensor,
        power_injection: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DC power flow constraint violation loss.
        
        Loss = ||P - B * θ||²
        
        Args:
            phase_angles: Predicted phase angles
            power_injection: Net power injection (load - generation)
            adjacency: Network adjacency matrix
            
        Returns:
            Constraint violation loss
        """
        batch_size, horizon, num_zones = phase_angles.shape
        
        # Compute expected power from angles: P = B * θ
        B = self.susceptance
        if adjacency is not None:
            mask = adjacency + torch.eye(num_zones, device=B.device)
            B = B * mask
        
        # Matrix multiplication: P_expected = B @ θ
        theta_flat = phase_angles.reshape(-1, num_zones)  # (batch*horizon, num_zones)
        P_expected_flat = torch.matmul(theta_flat, B.T)
        P_expected = P_expected_flat.reshape(batch_size, horizon, num_zones)
        
        # Violation is difference between actual and expected
        violation = power_injection - P_expected
        
        # Normalized MSE loss
        loss = (violation ** 2).mean() / (power_injection.abs().mean() + 1e-8)
        
        return loss
    
    def estimate_angles(
        self,
        load_pred: torch.Tensor,
        generation_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate phase angles from load and generation predictions.
        
        Args:
            load_pred: Predicted load at each bus
            generation_pred: Predicted generation at each bus
            
        Returns:
            Estimated phase angles
        """
        # Net injection = generation - load
        if generation_pred is not None:
            net_injection = generation_pred - load_pred
        else:
            # Assume generation balances load system-wide
            # Distribute slack proportionally
            total_load = load_pred.sum(dim=-1, keepdim=True)
            load_share = load_pred / (total_load + 1e-8)
            net_injection = -load_pred + load_share * total_load
        
        # Solve power flow
        return self.forward(net_injection)
    
    def _remove_reference_bus(self, matrix: torch.Tensor) -> torch.Tensor:
        """Remove reference bus row and column from matrix."""
        indices = [i for i in range(self.num_zones) if i != self.reference_bus]
        return matrix[indices][:, indices]
    
    def _remove_reference_bus_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Remove reference bus element from vector."""
        indices = [i for i in range(self.num_zones) if i != self.reference_bus]
        return vector[..., indices]
    
    def _insert_reference_bus(self, vector: torch.Tensor) -> torch.Tensor:
        """Insert zero at reference bus position."""
        batch_shape = vector.shape[:-1]
        full = torch.zeros(*batch_shape, self.num_zones, device=vector.device)
        
        indices = [i for i in range(self.num_zones) if i != self.reference_bus]
        full[..., indices] = vector
        
        return full


class LineFlowConstraint(nn.Module):
    """
    Transmission line flow constraint.
    
    Ensures predicted flows don't exceed line thermal limits.
    """
    
    def __init__(self, num_lines: int):
        super().__init__()
        self.num_lines = num_lines
        
        # Line thermal limits (learnable or fixed)
        self.thermal_limits = nn.Parameter(
            torch.ones(num_lines) * 1000,  # Default 1000 MW
            requires_grad=False
        )
        
    def forward(
        self,
        line_flows: torch.Tensor,
        soft_penalty: bool = True
    ) -> torch.Tensor:
        """
        Compute line flow constraint violation.
        
        Args:
            line_flows: Predicted line flows (batch, horizon, num_lines)
            soft_penalty: Use soft penalty (True) or hard constraint (False)
            
        Returns:
            Constraint violation loss
        """
        # Violation = max(0, |flow| - limit)
        violation = torch.relu(line_flows.abs() - self.thermal_limits)
        
        if soft_penalty:
            # Quadratic penalty
            loss = (violation ** 2).mean()
        else:
            # Count violations
            loss = (violation > 0).float().mean()
        
        return loss
