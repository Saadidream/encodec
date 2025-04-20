# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
import argparse
from pathlib import Path
from typing import Optional

from .modules.style_transfer import NeuralAudioStyleTransfer

def apply_style_transfer(
    model: NeuralAudioStyleTransfer,
    content_path: Path,
    style_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    device: Optional[str] = None
) -> None:
    """Apply style transfer to an audio file.
    
    Args:
        model: Trained style transfer model
        content_path: Path to content audio file
        style_path: Path to style reference audio file
        output_path: Path to save the output audio
        sample_rate: Target sample rate
        device: Device to run inference on
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load content audio
    content_audio, sr = torchaudio.load(content_path)
    if sr != sample_rate:
        content_audio = torchaudio.transforms.Resample(sr, sample_rate)(content_audio)
    
    # Load style audio
    style_audio, sr = torchaudio.load(style_path)
    if sr != sample_rate:
        style_audio = torchaudio.transforms.Resample(sr, sample_rate)(style_audio)
    
    # Move to device
    content_audio = content_audio.to(device)
    style_audio = style_audio.to(device)
    
    # Apply style transfer
    model.eval()
    with torch.no_grad():
        generated_audio = model(content_audio, style_audio)
    
    # Save output
    torchaudio.save(
        output_path,
        generated_audio.cpu(),
        sample_rate,
        encoding='PCM_S',
        bits_per_sample=16
    )

def main():
    parser = argparse.ArgumentParser(description="Apply Neural Audio Style Transfer")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--content-path", type=str, required=True, help="Path to content audio file")
    parser.add_argument("--style-path", type=str, required=True, help="Path to style reference audio file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save output audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = NeuralAudioStyleTransfer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Apply style transfer
    apply_style_transfer(
        model=model,
        content_path=Path(args.content_path),
        style_path=Path(args.style_path),
        output_path=Path(args.output_path),
        sample_rate=args.sample_rate,
        device=args.device
    )

if __name__ == "__main__":
    main() 
