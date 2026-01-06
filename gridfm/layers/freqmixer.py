"""
FreqMixer: Frequency-domain mixing layer for grid-specific pattern adaptation.

This layer transforms general-purpose foundation model representations into
power-grid-specific patterns by operating in the spectral domain with
grid-specific initialization.
"""

import torch
import torch.nn as nn
import torch.fft as fft
from typing import List, Optional, Tuple
import math


class FreqMixer(nn.Module):
    """
    Frequency-domain mixing layer that adapts foundation model outputs
    to power grid-specific temporal patterns.
    
    The FreqMixer operates by:
    1. Transforming input to frequency domain via FFT
    2. Applying learnable frequency-wise mixing weights
    3. Emphasizing grid-relevant frequencies (daily, weekly, yearly cycles)
    4. Transforming back to time domain
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden representation
        num_freq_components: Number of frequency components to retain
        delta: Frequency tolerance (±δ bins around target frequencies)
        grid_frequencies: List of target grid frequencies to emphasize
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_freq_components: int = 64,
        delta: int = 2,
        grid_frequencies: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freq_components = num_freq_components
        self.delta = delta
        
        # Default grid frequencies (cycles per sample at 5-min resolution)
        # Daily: 1/288, Weekly: 1/2016, Yearly: 1/105120
        self.grid_frequencies = grid_frequencies or [
            1/288,      # Daily cycle (288 samples/day)
            1/2016,     # Weekly cycle
            1/105120,   # Yearly cycle
            1/144,      # 12-hour cycle
            1/96,       # 8-hour cycle (business hours)
            1/48,       # 4-hour cycle
        ]
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Frequency-wise mixing weights (complex-valued via real/imag pairs)
        self.freq_weight_real = nn.Parameter(
            torch.randn(num_freq_components, hidden_dim) * 0.02
        )
        self.freq_weight_imag = nn.Parameter(
            torch.randn(num_freq_components, hidden_dim) * 0.02
        )
        
        # Grid frequency emphasis weights
        self.grid_emphasis = nn.Parameter(torch.ones(len(self.grid_frequencies)))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize with grid-specific priors
        self._init_grid_weights()
        
    def _init_grid_weights(self) -> None:
        """Initialize frequency weights with grid-specific priors."""
        # Enhance weights at grid-relevant frequencies
        for i, freq in enumerate(self.grid_frequencies):
            # Find corresponding frequency bin (approximate)
            freq_bin = int(freq * self.num_freq_components * 2)
            freq_bin = min(freq_bin, self.num_freq_components - 1)
            
            # Apply delta tolerance
            for d in range(-self.delta, self.delta + 1):
                bin_idx = freq_bin + d
                if 0 <= bin_idx < self.num_freq_components:
                    # Initialize with higher magnitude at grid frequencies
                    self.freq_weight_real.data[bin_idx] *= 2.0
                    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through FreqMixer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: Whether to return frequency attention weights
            
        Returns:
            output: Transformed tensor of shape (batch, hidden_dim)
            attention: Optional frequency attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Transform to frequency domain
        x_freq = fft.rfft(x, dim=1)  # (batch, freq_bins, hidden_dim)
        
        # Truncate to num_freq_components
        freq_bins = min(x_freq.shape[1], self.num_freq_components)
        x_freq = x_freq[:, :freq_bins, :]
        
        # Apply learnable frequency mixing
        weight_complex = torch.complex(
            self.freq_weight_real[:freq_bins],
            self.freq_weight_imag[:freq_bins]
        )
        
        # Element-wise multiplication in frequency domain
        x_mixed = x_freq * weight_complex.unsqueeze(0)
        
        # Apply grid frequency emphasis
        emphasis_weights = self._compute_emphasis_weights(freq_bins, seq_len)
        x_mixed = x_mixed * emphasis_weights.unsqueeze(0).unsqueeze(-1)
        
        # Transform back to time domain
        # Pad to original size for IRFFT
        x_padded = torch.zeros(
            batch_size, seq_len // 2 + 1, self.hidden_dim,
            dtype=x_mixed.dtype, device=x_mixed.device
        )
        x_padded[:, :freq_bins, :] = x_mixed
        
        x_time = fft.irfft(x_padded, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        
        # Global pooling to get fixed-size output
        x_pooled = x_time.mean(dim=1)  # (batch, hidden_dim)
        
        # Output projection and normalization
        output = self.output_proj(x_pooled)
        output = self.norm(output)
        
        # Compute attention weights if requested
        attention = None
        if return_attention:
            attention = self._compute_freq_attention(x_freq, weight_complex)
            
        return output, attention
    
    def _compute_emphasis_weights(
        self,
        freq_bins: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Compute emphasis weights for grid-relevant frequencies.
        
        Args:
            freq_bins: Number of frequency bins
            seq_len: Original sequence length
            
        Returns:
            Emphasis weights tensor of shape (freq_bins,)
        """
        # Frequency resolution
        freq_resolution = 1.0 / seq_len
        
        # Initialize weights
        weights = torch.ones(freq_bins, device=self.grid_emphasis.device)
        
        # Apply emphasis at grid frequencies
        for i, target_freq in enumerate(self.grid_frequencies):
            target_bin = int(target_freq / freq_resolution)
            
            # Apply Gaussian-weighted emphasis around target frequency
            for j in range(freq_bins):
                distance = abs(j - target_bin)
                if distance <= self.delta:
                    gaussian_weight = math.exp(-0.5 * (distance / self.delta) ** 2)
                    weights[j] += self.grid_emphasis[i] * gaussian_weight
        
        return weights
    
    def _compute_freq_attention(
        self,
        x_freq: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frequency attention scores for interpretability.
        
        Args:
            x_freq: Frequency domain input
            weights: Mixing weights
            
        Returns:
            Attention scores of shape (batch, freq_bins)
        """
        # Compute magnitude of contribution at each frequency
        contribution = (x_freq * weights.unsqueeze(0)).abs()
        
        # Sum across hidden dimension
        attention = contribution.sum(dim=-1)
        
        # Normalize
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        return attention.real if attention.is_complex() else attention


class FreqMixerBlock(nn.Module):
    """
    FreqMixer with residual connection and feed-forward network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_freq_components: int = 64,
        delta: int = 2,
        grid_frequencies: Optional[List[float]] = None,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.freqmixer = FreqMixer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_freq_components=num_freq_components,
            delta=delta,
            grid_frequencies=grid_frequencies
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connection."""
        # FreqMixer
        mixed, attention = self.freqmixer(x, return_attention)
        
        # Feed-forward with residual
        output = self.norm(mixed + self.ff(mixed))
        
        return output, attention
