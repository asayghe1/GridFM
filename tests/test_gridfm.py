"""
Unit tests for GridFM components.
"""

import pytest
import torch
import numpy as np

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFreqMixer:
    """Tests for FreqMixer layer."""
    
    def test_import(self):
        """Test that FreqMixer can be imported."""
        from gridfm.layers.freqmixer import FreqMixer
        assert FreqMixer is not None
    
    def test_output_shape(self):
        """Test FreqMixer output shape."""
        from gridfm.layers.freqmixer import FreqMixer
        
        freqmixer = FreqMixer(
            input_dim=64,
            hidden_dim=128,
            num_freq_components=32
        )
        
        batch_size = 2
        seq_len = 288
        input_dim = 64
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output, attention = freqmixer(x, return_attention=True)
        
        assert output.shape == (batch_size, 128)
        assert attention is not None
    
    def test_grid_frequencies(self):
        """Test custom grid frequencies."""
        from gridfm.layers.freqmixer import FreqMixer
        
        custom_freqs = [1/288, 1/2016]
        freqmixer = FreqMixer(
            input_dim=32,
            hidden_dim=64,
            grid_frequencies=custom_freqs
        )
        
        assert freqmixer.grid_frequencies == custom_freqs


class TestZonalGCN:
    """Tests for Zonal GCN."""
    
    def test_import(self):
        """Test that ZonalGCN can be imported."""
        from gridfm.layers.gcn import ZonalGCN
        assert ZonalGCN is not None
    
    def test_output_shape(self):
        """Test GCN output shape."""
        from gridfm.layers.gcn import ZonalGCN
        
        num_zones = 11
        gcn = ZonalGCN(
            input_dim=64,
            hidden_dim=32,
            output_dim=64,
            num_zones=num_zones
        )
        
        batch_size = 4
        x = torch.randn(batch_size, num_zones, 64)
        adjacency = torch.eye(num_zones)
        
        output, _ = gcn(x, adjacency)
        
        assert output.shape == (batch_size, num_zones, 64)


class TestPowerBalance:
    """Tests for power balance constraints."""
    
    def test_import(self):
        """Test that PowerBalanceLoss can be imported."""
        from gridfm.physics.power_balance import PowerBalanceLoss
        assert PowerBalanceLoss is not None
    
    def test_loss_computation(self):
        """Test power balance loss computation."""
        from gridfm.physics.power_balance import PowerBalanceLoss
        
        loss_fn = PowerBalanceLoss()
        
        batch_size = 2
        horizon = 24
        num_zones = 11
        
        load = torch.randn(batch_size, horizon, num_zones) * 1000 + 5000
        generation = load * 1.01  # Slightly more than load
        
        loss = loss_fn(load, generation)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestGridFMConfig:
    """Tests for GridFM configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        from gridfm import GridFMConfig
        
        config = GridFMConfig()
        
        assert config.hidden_dim == 256
        assert config.num_zones == 11
        assert "load" in config.tasks
    
    def test_custom_config(self):
        """Test custom configuration."""
        from gridfm import GridFMConfig
        
        config = GridFMConfig(
            hidden_dim=128,
            num_zones=5,
            tasks=["load", "lbmp"]
        )
        
        assert config.hidden_dim == 128
        assert config.num_zones == 5
        assert len(config.tasks) == 2
    
    def test_config_save_load(self, tmp_path):
        """Test config save and load."""
        from gridfm import GridFMConfig
        
        config = GridFMConfig(hidden_dim=512)
        config_path = tmp_path / "config.json"
        config.save(config_path)
        
        loaded_config = GridFMConfig.load(config_path)
        
        assert loaded_config.hidden_dim == 512


class TestGridFMModel:
    """Tests for main GridFM model."""
    
    def test_import(self):
        """Test that GridFM can be imported."""
        from gridfm import GridFM
        assert GridFM is not None
    
    def test_model_initialization(self):
        """Test model initialization."""
        from gridfm import GridFM, GridFMConfig
        
        config = GridFMConfig(
            hidden_dim=64,
            num_freq_components=16,
            num_zones=5,
            forecast_horizon=12
        )
        
        model = GridFM(config)
        
        assert model is not None
        assert len(list(model.parameters())) > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        from gridfm import GridFM, GridFMConfig
        
        config = GridFMConfig(
            hidden_dim=64,
            num_freq_components=16,
            num_zones=5,
            forecast_horizon=12,
            tasks=["load"]
        )
        
        model = GridFM(config)
        model.eval()
        
        batch_size = 2
        seq_len = 144
        num_zones = 5
        
        x = torch.randn(batch_size, seq_len, num_zones, 1)
        adjacency = torch.eye(num_zones)
        
        with torch.no_grad():
            predictions = model(x, adjacency)
        
        assert "load" in predictions
        assert predictions["load"].shape[0] == batch_size


class TestDataLoader:
    """Tests for NYISO data loader."""
    
    def test_import(self):
        """Test that NYISODataLoader can be imported."""
        from gridfm.data.nyiso import NYISODataLoader
        assert NYISODataLoader is not None
    
    def test_zone_mapping(self):
        """Test zone mapping."""
        from gridfm.data.nyiso import NYISO_ZONES, ZONE_TO_IDX
        
        assert len(NYISO_ZONES) == 11
        assert ZONE_TO_IDX["N.Y.C."] == 9
    
    def test_adjacency_matrix(self):
        """Test adjacency matrix generation."""
        from gridfm.data.nyiso import NYISODataLoader
        
        loader = NYISODataLoader(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        adjacency = loader.get_adjacency_matrix()
        
        assert adjacency.shape == (11, 11)
        assert adjacency.sum() > 0  # Has connections
        assert (adjacency == adjacency.T).all()  # Symmetric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
