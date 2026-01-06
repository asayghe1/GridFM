"""
GridFM: Main model architecture combining foundation model backbone with
physics-informed constraints and multi-task learning.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import json

from gridfm.layers.freqmixer import FreqMixer
from gridfm.layers.gcn import ZonalGCN
from gridfm.models.backbone import load_backbone
from gridfm.models.task_heads import MultiTaskHead
from gridfm.physics.power_balance import PowerBalanceLoss
from gridfm.physics.dc_power_flow import DCPowerFlowConstraint


@dataclass
class GridFMConfig:
    """Configuration for GridFM model."""
    
    # Backbone configuration
    backbone: str = "moirai-moe-base"
    backbone_dim: int = 512
    freeze_backbone: bool = True
    
    # FreqMixer configuration
    hidden_dim: int = 256
    num_freq_components: int = 64
    freq_delta: int = 2
    grid_frequencies: List[float] = field(default_factory=lambda: [
        1/288, 1/2016, 1/8760,  # Daily, weekly, yearly
        1/144, 1/96, 1/48       # 12h, 8h, 4h cycles
    ])
    
    # GCN configuration
    num_zones: int = 11
    gcn_hidden_dim: int = 128
    gcn_num_layers: int = 2
    
    # Task configuration
    tasks: List[str] = field(default_factory=lambda: [
        "load", "lbmp", "emissions", "renewable"
    ])
    forecast_horizon: int = 24
    
    # Physics constraints
    enable_power_balance: bool = True
    enable_dc_power_flow: bool = True
    physics_weight: float = 0.1
    coupling_weight: float = 0.05
    
    # Training
    dropout: float = 0.1
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            k: v if not isinstance(v, list) else list(v)
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GridFMConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "GridFMConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class GridFM(nn.Module):
    """
    GridFM: Physics-Informed Foundation Model for Multi-Task Energy Forecasting.
    
    This model combines:
    1. Pre-trained time series foundation model backbone (e.g., Moirai-MoE)
    2. FreqMixer adaptation layer for grid-specific frequency patterns
    3. Graph Neural Network for zonal topology encoding
    4. Multi-task output heads with uncertainty weighting
    5. Physics-informed constraints (power balance, DC power flow)
    
    Args:
        config: GridFMConfig instance with model hyperparameters
    """
    
    def __init__(self, config: GridFMConfig):
        super().__init__()
        self.config = config
        
        # Load backbone foundation model
        self.backbone = load_backbone(
            config.backbone,
            freeze=config.freeze_backbone
        )
        
        # FreqMixer adaptation layer
        self.freqmixer = FreqMixer(
            input_dim=config.backbone_dim,
            hidden_dim=config.hidden_dim,
            num_freq_components=config.num_freq_components,
            delta=config.freq_delta,
            grid_frequencies=config.grid_frequencies
        )
        
        # Zonal GCN for spatial dependencies
        self.zonal_gcn = ZonalGCN(
            input_dim=config.hidden_dim,
            hidden_dim=config.gcn_hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.gcn_num_layers,
            num_zones=config.num_zones
        )
        
        # Multi-task output heads
        self.task_heads = MultiTaskHead(
            input_dim=config.hidden_dim,
            tasks=config.tasks,
            forecast_horizon=config.forecast_horizon,
            num_zones=config.num_zones
        )
        
        # Physics constraint modules
        if config.enable_power_balance:
            self.power_balance_loss = PowerBalanceLoss()
        if config.enable_dc_power_flow:
            self.dc_power_flow = DCPowerFlowConstraint(
                num_zones=config.num_zones
            )
        
        # Uncertainty weights for multi-task learning (learnable)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in config.tasks
        })
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GridFM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_zones, features)
            adjacency: Adjacency matrix of shape (num_zones, num_zones)
            timestamps: Optional timestamps for temporal encoding
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - Task predictions (load, lbmp, emissions, renewable)
                - Phase angles (if DC power flow enabled)
                - Attention weights (if requested)
        """
        batch_size, seq_len, num_zones, features = x.shape
        
        # Reshape for backbone: (batch * zones, seq_len, features)
        x_flat = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_zones, seq_len, features
        )
        
        # Pass through backbone
        backbone_out = self.backbone(x_flat, timestamps)
        
        # Apply FreqMixer adaptation
        freq_out, freq_attention = self.freqmixer(
            backbone_out, 
            return_attention=True
        )
        
        # Reshape back: (batch, num_zones, hidden_dim)
        freq_out = freq_out.reshape(batch_size, num_zones, -1)
        
        # Apply zonal GCN for spatial dependencies
        gcn_out, gcn_attention = self.zonal_gcn(
            freq_out, 
            adjacency,
            return_attention=True
        )
        
        # Apply dropout
        gcn_out = self.dropout(gcn_out)
        
        # Generate task-specific predictions
        predictions = self.task_heads(gcn_out)
        
        # Add phase angles if DC power flow enabled
        if self.config.enable_dc_power_flow:
            predictions["phase_angles"] = self.dc_power_flow.estimate_angles(
                predictions.get("load"),
                predictions.get("renewable")
            )
        
        # Optionally return attention weights
        if return_attention:
            predictions["freq_attention"] = freq_attention
            predictions["gcn_attention"] = gcn_attention
            
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss with physics constraints.
        
        Args:
            predictions: Model predictions for each task
            targets: Ground truth values for each task
            adjacency: Adjacency matrix for power flow constraints
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Task-specific losses with uncertainty weighting
        for task in self.config.tasks:
            if task in predictions and task in targets:
                task_loss = nn.functional.mse_loss(
                    predictions[task], 
                    targets[task]
                )
                
                # Uncertainty weighting: loss / (2 * var) + log(var)
                precision = torch.exp(-self.log_vars[task])
                weighted_loss = precision * task_loss + self.log_vars[task]
                
                loss_dict[f"{task}_loss"] = task_loss.item()
                total_loss = total_loss + weighted_loss
        
        # Physics constraint losses
        if self.config.enable_power_balance:
            pb_loss = self.power_balance_loss(
                predictions.get("load"),
                predictions.get("renewable"),
                predictions.get("phase_angles"),
                adjacency
            )
            loss_dict["power_balance_loss"] = pb_loss.item()
            total_loss = total_loss + self.config.physics_weight * pb_loss
        
        if self.config.enable_dc_power_flow:
            dc_loss = self.dc_power_flow.compute_loss(
                predictions.get("phase_angles"),
                predictions.get("load"),
                adjacency
            )
            loss_dict["dc_power_flow_loss"] = dc_loss.item()
            total_loss = total_loss + self.config.physics_weight * dc_loss
        
        # Adaptive coupling loss between tasks
        if "load" in predictions and "lbmp" in predictions:
            coupling_loss = self._compute_coupling_loss(
                predictions["load"],
                predictions["lbmp"]
            )
            loss_dict["coupling_loss"] = coupling_loss.item()
            total_loss = total_loss + self.config.coupling_weight * coupling_loss
        
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_coupling_loss(
        self,
        load_pred: torch.Tensor,
        price_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive coupling loss between load and price predictions.
        
        Encourages the model to learn correlations between load and price.
        """
        # Normalize predictions
        load_norm = (load_pred - load_pred.mean()) / (load_pred.std() + 1e-8)
        price_norm = (price_pred - price_pred.mean()) / (price_pred.std() + 1e-8)
        
        # Correlation-based coupling loss
        correlation = (load_norm * price_norm).mean()
        
        # We expect positive correlation (higher load -> higher prices)
        # Loss encourages learning this relationship
        coupling_loss = 1 - correlation.abs()
        
        return coupling_loss
    
    def predict(
        self,
        dataloader,
        horizon: int = 24,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for given data.
        
        Args:
            dataloader: DataLoader with input sequences
            horizon: Forecast horizon (number of steps ahead)
            tasks: List of tasks to predict (default: all)
            
        Returns:
            Dictionary of predictions for each task
        """
        self.eval()
        tasks = tasks or self.config.tasks
        
        all_predictions = {task: [] for task in tasks}
        all_targets = {task: [] for task in tasks}
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch["input"]
                adjacency = batch["adjacency"]
                
                predictions = self.forward(x, adjacency)
                
                for task in tasks:
                    if task in predictions:
                        all_predictions[task].append(predictions[task].cpu())
                    if task in batch:
                        all_targets[task].append(batch[task].cpu())
        
        # Concatenate all batches
        results = {
            "predictions": {
                task: torch.cat(preds, dim=0)
                for task, preds in all_predictions.items()
            },
            "targets": {
                task: torch.cat(tgts, dim=0)
                for task, tgts in all_targets.items()
                if tgts
            }
        }
        
        # Compute metrics
        results["metrics"] = self._compute_metrics(
            results["predictions"],
            results["targets"]
        )
        
        return results
    
    def _compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute evaluation metrics for each task."""
        metrics = {}
        
        for task in predictions:
            if task in targets:
                pred = predictions[task]
                tgt = targets[task]
                
                # MAPE (excluding near-zero values for prices)
                mask = tgt.abs() > 1.0 if task == "lbmp" else torch.ones_like(tgt, dtype=bool)
                if mask.sum() > 0:
                    mape = (((pred - tgt).abs() / (tgt.abs() + 1e-8)) * mask).sum() / mask.sum() * 100
                    metrics[f"{task}_mape"] = mape.item()
                
                # RMSE
                rmse = torch.sqrt(nn.functional.mse_loss(pred, tgt))
                metrics[f"{task}_rmse"] = rmse.item()
                
                # MAE
                mae = nn.functional.l1_loss(pred, tgt)
                metrics[f"{task}_mae"] = mae.item()
        
        return metrics
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> "GridFM":
        """
        Load a pre-trained GridFM model.
        
        Args:
            model_name_or_path: Model identifier or path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded GridFM model
        """
        path = Path(model_name_or_path)
        
        if path.exists():
            # Load from local path
            config_path = path / "config.json"
            weights_path = path / "model.pt"
        else:
            # Download from hub (placeholder)
            raise NotImplementedError(
                f"Downloading from hub not yet implemented. "
                f"Please provide local path to model."
            )
        
        config = GridFMConfig.load(config_path)
        model = cls(config)
        
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        return model
    
    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Save model and config to directory.
        
        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(path / "config.json")
        
        # Save model weights
        torch.save(self.state_dict(), path / "model.pt")
        
        print(f"Model saved to {path}")
