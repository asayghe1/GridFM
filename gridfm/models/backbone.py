"""
Backbone foundation model loader.

Supports loading various pre-trained time series foundation models.
"""

import torch
import torch.nn as nn
from typing import Optional


def load_backbone(
    model_name: str,
    freeze: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Load a pre-trained backbone foundation model.
    
    Supported models:
    - moirai-moe-base: Moirai Mixture-of-Experts model
    - moirai-moe-large: Large Moirai MoE model
    - chronos-base: Amazon Chronos model
    - timesfm: Google TimesFM model
    
    Args:
        model_name: Name of the model to load
        freeze: Whether to freeze backbone weights
        device: Device to load model on
        
    Returns:
        Loaded backbone model
    """
    if model_name.startswith("moirai"):
        backbone = load_moirai(model_name, device)
    elif model_name.startswith("chronos"):
        backbone = load_chronos(model_name, device)
    elif model_name.startswith("timesfm"):
        backbone = load_timesfm(model_name, device)
    else:
        # Default: use a simple transformer encoder as placeholder
        backbone = SimpleTransformerBackbone()
    
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
    
    return backbone.to(device)


def load_moirai(model_name: str, device: str) -> nn.Module:
    """Load Moirai MoE model."""
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        
        # Map model name to checkpoint
        checkpoint_map = {
            "moirai-moe-base": "Salesforce/moirai-moe-1.0-R-base",
            "moirai-moe-large": "Salesforce/moirai-moe-1.0-R-large",
            "moirai-moe-small": "Salesforce/moirai-moe-1.0-R-small",
        }
        
        checkpoint = checkpoint_map.get(model_name, model_name)
        
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(checkpoint),
            prediction_length=24,
            context_length=288,
            patch_size=32,
            num_samples=100,
        )
        
        return MoiraiWrapper(model)
        
    except ImportError:
        print("Warning: uni2ts not installed. Using placeholder backbone.")
        return SimpleTransformerBackbone()


def load_chronos(model_name: str, device: str) -> nn.Module:
    """Load Amazon Chronos model."""
    try:
        from chronos import ChronosPipeline
        
        checkpoint_map = {
            "chronos-base": "amazon/chronos-t5-base",
            "chronos-large": "amazon/chronos-t5-large",
            "chronos-small": "amazon/chronos-t5-small",
        }
        
        checkpoint = checkpoint_map.get(model_name, model_name)
        
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint,
            device_map=device,
            torch_dtype=torch.float32,
        )
        
        return ChronosWrapper(pipeline)
        
    except ImportError:
        print("Warning: chronos not installed. Using placeholder backbone.")
        return SimpleTransformerBackbone()


def load_timesfm(model_name: str, device: str) -> nn.Module:
    """Load Google TimesFM model."""
    try:
        import timesfm
        
        tfm = timesfm.TimesFm(
            context_len=288,
            horizon_len=24,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        
        return TimesFMWrapper(tfm)
        
    except ImportError:
        print("Warning: timesfm not installed. Using placeholder backbone.")
        return SimpleTransformerBackbone()


class MoiraiWrapper(nn.Module):
    """Wrapper for Moirai model to standardize interface."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_dim = 512  # Moirai hidden dimension
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Moirai.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            timestamps: Optional timestamp tensor
            
        Returns:
            Hidden representations (batch, seq_len, hidden_dim)
        """
        # Moirai expects specific input format
        # This is a simplified wrapper
        return self.model.module.encode(x)


class ChronosWrapper(nn.Module):
    """Wrapper for Chronos model."""
    
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.output_dim = 768  # T5-base hidden dimension
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Chronos encoder."""
        # Get encoder hidden states
        return self.pipeline.model.encoder(x).last_hidden_state


class TimesFMWrapper(nn.Module):
    """Wrapper for TimesFM model."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_dim = 1280  # TimesFM hidden dimension
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through TimesFM encoder."""
        return self.model.encode(x)


class SimpleTransformerBackbone(nn.Module):
    """
    Simple transformer encoder as fallback backbone.
    
    Used when pre-trained models are not available.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            timestamps: Optional timestamps (unused in simple version)
            
        Returns:
            Hidden representations (batch, seq_len, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
