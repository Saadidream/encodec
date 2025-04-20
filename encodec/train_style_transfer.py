# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np

from .modules.style_transfer import NeuralAudioStyleTransfer

class AudioStyleDataset(Dataset):
    """Dataset for audio style transfer training."""
    def __init__(self, content_dir: Path, style_dir: Path, sample_rate: int = 16000):
        self.content_files = list(content_dir.glob("*.wav"))
        self.style_files = list(style_dir.glob("*.wav"))
        self.sample_rate = sample_rate
        
    def __len__(self) -> int:
        return len(self.content_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load content audio
        content_path = self.content_files[idx]
        content_audio, sr = torchaudio.load(content_path)
        if sr != self.sample_rate:
            content_audio = torchaudio.transforms.Resample(sr, self.sample_rate)(content_audio)
        
        # Load random style audio
        style_path = np.random.choice(self.style_files)
        style_audio, sr = torchaudio.load(style_path)
        if sr != self.sample_rate:
            style_audio = torchaudio.transforms.Resample(sr, self.sample_rate)(style_audio)
        
        return content_audio, style_audio

def train(args):
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAudioStyleTransfer().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create dataset and dataloader
    dataset = AudioStyleDataset(
        content_dir=Path(args.content_dir),
        style_dir=Path(args.style_dir),
        sample_rate=args.sample_rate
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (content_audio, style_audio) in enumerate(dataloader):
            content_audio = content_audio.to(device)
            style_audio = style_audio.to(device)
            
            # Forward pass
            generated_audio = model(content_audio, style_audio)
            
            # Compute losses
            style_loss = model.compute_style_loss(generated_audio, style_audio)
            content_loss = model.compute_content_loss(generated_audio, content_audio)
            reconstruction_loss = nn.L1Loss()(generated_audio, content_audio)
            
            # Total loss
            loss = (
                args.style_weight * style_loss +
                args.content_weight * content_loss +
                args.reconstruction_weight * reconstruction_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="Train Neural Audio Style Transfer model")
    
    # Data parameters
    parser.add_argument("--content-dir", type=str, required=True, help="Directory containing content audio files")
    parser.add_argument("--style-dir", type=str, required=True, help="Directory containing style audio files")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Loss weights
    parser.add_argument("--style-weight", type=float, default=1.0, help="Weight for style loss")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Weight for content loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="Weight for reconstruction loss")
    
    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in batches")
    parser.add_argument("--save-interval", type=int, default=5, help="Checkpoint saving interval in epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main() 
