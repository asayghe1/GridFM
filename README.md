# GridFM: A Physics-Informed Foundation Model for Multi-Task Energy Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/)

<p align="center">
  <img src="docs/images/gridfm_architecture.png" alt="GridFM Architecture" width="800"/>
</p>

**GridFM** is a physics-informed foundation model designed for multi-task energy forecasting using real-time data from Independent System Operators (ISOs). It combines pre-trained time series foundation models with domain-specific adaptations that respect power system physics.

## 🌟 Key Features

- **FreqMixer Adaptation Layer**: Novel frequency-domain mixing mechanism that transforms general-purpose foundation model representations into power-grid-specific patterns
- **Physics-Informed Constraints**: Integration of power balance equations and DC power flow approximations
- **Multi-Task Learning**: Simultaneous forecasting of load, prices, emissions, and renewable generation
- **Graph Neural Networks**: Zonal topology encoding via GCN for spatial dependencies
- **Explainability**: SHAP-based feature attribution and attention visualization

## 📊 Performance

GridFM achieves state-of-the-art performance on NYISO, PJM, and CAISO datasets:

| Task | MAPE | Improvement vs. Fine-tuned Baseline |
|------|------|-------------------------------------|
| Load Forecasting | 2.14% ± 0.05% | 10.1% |
| Price Forecasting | 7.80% ± 0.31% | 15.9% |
| Emission Prediction | 4.73% ± 0.18% | 14.3% |
| Renewable Generation | 8.92% ± 0.42% | 12.7% |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GridFM/GridFM.git
cd GridFM

# Create conda environment
conda create -n gridfm python=3.10
conda activate gridfm

# Install dependencies
pip install -r requirements.txt

# Install GridFM
pip install -e .
```

### Basic Usage

```python
from gridfm import GridFM
from gridfm.data import NYISODataLoader

# Load pre-trained model
model = GridFM.from_pretrained("gridfm-base-nyiso")

# Load data
dataloader = NYISODataLoader(
    start_date="2023-01-01",
    end_date="2023-12-31",
    zones=["WEST", "CENTRL", "NORTH"]
)

# Generate forecasts
forecasts = model.predict(
    dataloader,
    horizon=24,  # 24 steps ahead (2 hours at 5-min resolution)
    tasks=["load", "lbmp", "emissions"]
)

# Access results
print(f"Load MAPE: {forecasts.metrics['load_mape']:.2f}%")
```

### Training from Scratch

```python
from gridfm import GridFM, GridFMConfig
from gridfm.training import Trainer

# Configure model
config = GridFMConfig(
    backbone="moirai-moe-base",
    num_zones=11,
    hidden_dim=256,
    num_freq_components=64,
    physics_weight=0.1,
    coupling_weight=0.05
)

# Initialize model
model = GridFM(config)

# Train
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=1e-4,
    num_epochs=100
)
trainer.train()
```

## 📁 Repository Structure

```
GridFM/
├── gridfm/                    # Main package
│   ├── models/               # Model architectures
│   │   ├── gridfm.py        # Main GridFM model
│   │   ├── backbone.py      # Foundation model backbone
│   │   └── task_heads.py    # Task-specific output heads
│   ├── layers/              # Custom layers
│   │   ├── freqmixer.py     # FreqMixer adaptation layer
│   │   ├── gcn.py           # Graph convolutional network
│   │   └── attention.py     # Attention mechanisms
│   ├── physics/             # Physics constraints
│   │   ├── power_balance.py # Power balance equations
│   │   └── dc_power_flow.py # DC power flow approximation
│   ├── data/                # Data loading utilities
│   │   ├── nyiso.py         # NYISO data loader
│   │   ├── pjm.py           # PJM data loader
│   │   └── caiso.py         # CAISO data loader
│   └── utils/               # Utility functions
├── configs/                  # Configuration files
├── scripts/                  # Training and evaluation scripts
├── examples/                 # Example notebooks
├── docs/                     # Documentation
├── paper/                    # Paper and supplementary materials
└── tests/                    # Unit tests
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Data Preparation](docs/data_preparation.md)
- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api_reference.md)

## 🔧 Configuration

GridFM uses YAML configuration files. Example:

```yaml
# configs/gridfm_nyiso.yaml
model:
  backbone: moirai-moe-base
  hidden_dim: 256
  num_freq_components: 64
  freq_delta: 2
  
physics:
  enable_power_balance: true
  enable_dc_power_flow: true
  physics_weight: 0.1
  
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  early_stopping_patience: 10
```

## 📈 Reproducing Paper Results

To reproduce the results from our paper:

```bash
# Download and preprocess data
python scripts/download_data.py --iso nyiso --start 2019-01-01 --end 2023-12-31

# Train model
python scripts/train.py --config configs/gridfm_nyiso.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test_year 2023

# Generate figures
python scripts/generate_figures.py --results results/evaluation/
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_freqmixer.py -v

# Run with coverage
pytest --cov=gridfm tests/
```

## 📝 Citation

If you use GridFM in your research, please cite our paper:

```bibtex
@article{gridfm2024,
  title={GridFM: A Physics-Informed Foundation Model for Multi-Task Energy Forecasting Using Real-Time NYISO Data},
  author={Sayghe, Ali},
  journal={MDPI Energies},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NYISO, PJM, and CAISO for providing open access to grid data
- The Moirai team for the foundation model backbone
- PyTorch Geometric team for GNN implementations

## 📧 Contact

For questions or feedback, please open an issue or contact [sayghea@rcjy.edu.sa].

---

<p align="center">
  Made with ❤️ for sustainable energy systems
</p>
