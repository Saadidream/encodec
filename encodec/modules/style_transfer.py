# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class StyleEncoder(nn.Module):
    """Encoder network for extracting style features from audio.
    
    This module uses a combination of convolutional layers and attention
    to extract style-specific features from audio input.
    """
    def __init__(self, input_channels: int = 1, style_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.attention = nn.MultiheadAttention(256, num_heads=8)
        self.style_projection = nn.Linear(256, style_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, time]
        features = self.conv_layers(x)
        
        # Reshape for attention
        features = features.permute(2, 0, 1)  # [time, batch, channels]
        features, _ = self.attention(features, features, features)
        
        # Global average pooling
        style = features.mean(dim=0)  # [batch, channels]
        style = self.style_projection(style)
        return style

class ContentEncoder(nn.Module):
    """Encoder network for extracting content features from audio.
    
    This module focuses on capturing the structural and content-related
    features of the audio while being style-invariant.
    """
    def __init__(self, input_channels: int = 1, content_dim: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
        )
        
        self.content_projection = nn.Linear(256, content_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        # Global average pooling
        content = features.mean(dim=2)  # [batch, channels]
        content = self.content_projection(content)
        return content

class AudioDecoder(nn.Module):
    """Decoder network for generating audio from content and style features.
    
    This module combines content and style features to generate
    stylized audio output.
    """
    def __init__(self, content_dim: int = 256, style_dim: int = 256, output_channels: int = 1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(content_dim + style_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, output_channels, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
        )
        
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Combine content and style
        combined = torch.cat([content, style], dim=1)
        features = self.fusion(combined)
        
        # Reshape for deconvolution
        features = features.unsqueeze(2)  # Add time dimension
        audio = self.deconv_layers(features)
        return audio

class NeuralAudioStyleTransfer(nn.Module):
    """Complete neural audio style transfer model.
    
    This model can transfer the style of one audio to another while
    maintaining the content structure.
    """
    def __init__(self, input_channels: int = 1, content_dim: int = 256, style_dim: int = 256):
        super().__init__()
        self.content_encoder = ContentEncoder(input_channels, content_dim)
        self.style_encoder = StyleEncoder(input_channels, style_dim)
        self.decoder = AudioDecoder(content_dim, style_dim, input_channels)
        
    def encode_content(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features from input audio."""
        return self.content_encoder(x)
    
    def encode_style(self, x: torch.Tensor) -> torch.Tensor:
        """Extract style features from reference audio."""
        return self.style_encoder(x)
    
    def forward(self, content_audio: torch.Tensor, style_audio: torch.Tensor) -> torch.Tensor:
        """Transfer style from style_audio to content_audio."""
        content_features = self.encode_content(content_audio)
        style_features = self.encode_style(style_audio)
        return self.decoder(content_features, style_features)
    
    def compute_style_loss(self, generated: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Compute style loss between generated and style audio."""
        gen_features = self.style_encoder(generated)
        style_features = self.style_encoder(style)
        return F.mse_loss(gen_features, style_features)
    
    def compute_content_loss(self, generated: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """Compute content loss between generated and content audio."""
        gen_features = self.content_encoder(generated)
        content_features = self.content_encoder(content)
        return F.mse_loss(gen_features, content_features) 
