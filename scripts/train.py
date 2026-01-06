"""
Training script for GridFM.

Usage:
    python scripts/train.py --config configs/gridfm_nyiso.yaml
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridfm import GridFM, GridFMConfig
from gridfm.data.nyiso import NYISODataLoader


def main():
    parser = argparse.ArgumentParser(description="Train GridFM model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loaders
    print("Loading data...")
    data_config = config.get("data", {})
    
    train_loader = create_dataloader(
        start_date=data_config.get("train_start", "2019-01-01"),
        end_date=data_config.get("train_end", "2022-12-31"),
        batch_size=config.get("training", {}).get("batch_size", 32),
        sequence_length=data_config.get("sequence_length", 288),
        forecast_horizon=data_config.get("forecast_horizon", 24),
    )
    
    val_loader = create_dataloader(
        start_date=data_config.get("val_start", "2023-01-01"),
        end_date=data_config.get("val_end", "2023-06-30"),
        batch_size=config.get("training", {}).get("batch_size", 32),
        sequence_length=data_config.get("sequence_length", 288),
        forecast_horizon=data_config.get("forecast_horizon", 24),
    )
    
    # Initialize model
    print("Initializing model...")
    model_config = GridFMConfig(
        backbone=config.get("model", {}).get("backbone", "moirai-moe-base"),
        hidden_dim=config.get("model", {}).get("hidden_dim", 256),
        num_freq_components=config.get("model", {}).get("num_freq_components", 64),
        freq_delta=config.get("model", {}).get("freq_delta", 2),
        num_zones=config.get("model", {}).get("num_zones", 11),
        tasks=config.get("model", {}).get("tasks", ["load", "lbmp", "emissions", "renewable"]),
        forecast_horizon=data_config.get("forecast_horizon", 24),
        enable_power_balance=config.get("physics", {}).get("enable_power_balance", True),
        enable_dc_power_flow=config.get("physics", {}).get("enable_dc_power_flow", True),
        physics_weight=config.get("physics", {}).get("physics_weight", 0.1),
        coupling_weight=config.get("physics", {}).get("coupling_weight", 0.05),
    )
    
    model = GridFM(model_config).to(device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
    
    # Setup optimizer and scheduler
    training_config = config.get("training", {})
    
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    
    num_epochs = training_config.get("num_epochs", 100)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_config.get("early_stopping_patience", 10)
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                output_dir / "best_model.pt"
            )
            print("  Saved best model!")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save final model
    model.save_pretrained(output_dir / "final_model")
    print(f"Training complete! Model saved to {output_dir}")


def create_dataloader(
    start_date: str,
    end_date: str,
    batch_size: int,
    sequence_length: int,
    forecast_horizon: int,
    data_dir: str = "./data/nyiso"
) -> DataLoader:
    """Create data loader for given date range."""
    nyiso_loader = NYISODataLoader(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Download data if not exists
    nyiso_loader.download_data(force=False)
    
    # Create dataset
    dataset = nyiso_loader.create_dataset(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        x = batch["input"].to(device)
        adjacency = batch["adjacency"][0].to(device)  # Same for all samples
        
        targets = {
            key: batch[key].to(device)
            for key in ["load", "lbmp", "emissions", "renewable"]
            if key in batch
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x.unsqueeze(-1), adjacency)  # Add feature dim
        
        # Compute loss
        loss, loss_dict = model.compute_loss(predictions, targets, adjacency)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = {task: [] for task in model.config.tasks}
    all_targets = {task: [] for task in model.config.tasks}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            x = batch["input"].to(device)
            adjacency = batch["adjacency"][0].to(device)
            
            targets = {
                key: batch[key].to(device)
                for key in ["load", "lbmp", "emissions", "renewable"]
                if key in batch
            }
            
            predictions = model(x.unsqueeze(-1), adjacency)
            loss, _ = model.compute_loss(predictions, targets, adjacency)
            
            total_loss += loss.item()
            
            for task in model.config.tasks:
                if task in predictions and task in targets:
                    all_predictions[task].append(predictions[task].cpu())
                    all_targets[task].append(targets[task].cpu())
    
    # Compute metrics
    metrics = {}
    for task in model.config.tasks:
        if all_predictions[task]:
            preds = torch.cat(all_predictions[task], dim=0)
            tgts = torch.cat(all_targets[task], dim=0)
            
            # MAPE
            mask = tgts.abs() > 0.1
            if mask.sum() > 0:
                mape = ((preds - tgts).abs() / (tgts.abs() + 1e-8) * mask).sum() / mask.sum() * 100
                metrics[f"{task}_mape"] = mape.item()
            
            # RMSE
            rmse = torch.sqrt(nn.functional.mse_loss(preds, tgts))
            metrics[f"{task}_rmse"] = rmse.item()
    
    return total_loss / len(dataloader), metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path
) -> None:
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


if __name__ == "__main__":
    main()
