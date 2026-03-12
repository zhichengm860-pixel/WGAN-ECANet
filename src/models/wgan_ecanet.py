#!/usr/bin/env python3
"""
WGAN-ECANet Model Architecture
Aligned with paper description
- Multi-scale feature extraction: Parallel kernels (1x1, 3x3, 5x5, 7x7)
- Dimension-free ECANet: Single 1D convolution
- Dynamic gradient penalty: Three-stage strategy
- Spectral normalization: Stabilize GAN training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
from typing import Dict, Optional


class ECANet(nn.Module):
    """Dimension-free ECANet channel attention using 1D convolution without dimensionality reduction"""
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        self.channels = channels
        
        k = int(abs((math.log(channels, 2) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1
        self.kernel_size = max(3, k)
        
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, 
                              padding=(self.kernel_size - 1) // 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool1d(x, 1)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        attention = torch.sigmoid(y.transpose(-1, -2))
        return x * attention


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor using parallel kernels (1x1, 3x3, 5x5, 7x7)"""
    
    def __init__(self, in_channels: int, out_channels: int, use_multi_scale: bool = True):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        
        if use_multi_scale:
            self.conv1x1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, padding=0)
            self.conv3x3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
            self.conv5x5 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
            self.conv7x7 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        else:
            self.single_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_multi_scale:
            f1 = self.conv1x1(x)
            f3 = self.conv3x3(x)
            f5 = self.conv5x5(x)
            f7 = self.conv7x7(x)
            out = torch.cat([f1, f3, f5, f7], dim=1)
        else:
            out = self.single_conv(x)
        
        out = self.bn(out)
        out = self.relu(out)
        return out


class DynamicGradientPenalty:
    """Dynamic gradient penalty strategy with three-stage piecewise linear adjustment"""
    
    def __init__(self, base_lambda: float = 10.0, min_lambda: float = 5.0, use_dynamic: bool = True):
        self.base_lambda = base_lambda
        self.min_lambda = min_lambda
        self.use_dynamic = use_dynamic
        self.current_lambda = base_lambda
        self.total_steps = 0
        self.current_step = 0
        
    def set_total_steps(self, total_steps: int):
        self.total_steps = total_steps
        
    def step(self):
        self.current_step += 1
        if self.use_dynamic:
            self._update_lambda()
        
    def _update_lambda(self):
        if self.total_steps == 0:
            return
            
        t = self.current_step
        T = self.total_steps
        
        if t < 0.3 * T:
            self.current_lambda = self.base_lambda
        elif t < 0.7 * T:
            progress = (t - 0.3 * T) / (0.4 * T)
            self.current_lambda = self.base_lambda - 5 * progress
        else:
            self.current_lambda = self.min_lambda
        
    def compute_penalty(self, discriminator, real_samples, fake_samples, device):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
    
    def get_lambda(self) -> float:
        return self.current_lambda


class Generator(nn.Module):
    """WGAN Generator with ConvTranspose1d for signal generation
    
    Architecture:
    - Linear projection to initial feature map
    - Multiple ConvTranspose1d layers for upsampling
    - ECA attention for channel refinement
    - Output: (batch, channels, signal_length)
    """
    
    def __init__(self, noise_dim: int = 100, signal_length: int = 1024, 
                 num_classes: int = 24, channels: int = 2, use_eca: bool = True):
        super().__init__()
        self.noise_dim = noise_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        self.use_eca = use_eca
        
        # Calculate initial size: signal_length needs to be reached by repeated doubling
        # 1024 = 4 * 2^8, so we start from length 4
        self.init_length = 4
        self.init_channels = 512
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 50)
        
        # Project noise + class embedding to initial feature map
        # Input: noise(100) + class_emb(50) = 150
        # Output: 512 channels * 4 length = 2048
        self.project = nn.Sequential(
            nn.Linear(noise_dim + 50, self.init_channels * self.init_length),
            nn.BatchNorm1d(self.init_channels * self.init_length),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers with ConvTranspose1d
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 (8 layers)
        self.up1 = self._make_upsample_block(self.init_channels, 256)  # 4 -> 8
        self.up2 = self._make_upsample_block(256, 128)                  # 8 -> 16
        self.up3 = self._make_upsample_block(128, 64)                   # 16 -> 32
        self.up4 = self._make_upsample_block(64, 32)                    # 32 -> 64
        self.up5 = self._make_upsample_block(32, 16)                    # 64 -> 128
        self.up6 = self._make_upsample_block(16, 8)                     # 128 -> 256
        self.up7 = self._make_upsample_block(8, 4)                      # 256 -> 512
        
        # Final layer: 512 -> 2 channels
        self.final = nn.Sequential(
            nn.ConvTranspose1d(4, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _make_upsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block with ConvTranspose1d, BatchNorm, ReLU, and optional ECA"""
        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if self.use_eca:
            layers.append(ECANet(out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Concatenate noise and class embedding
        class_emb = self.class_embedding(labels)
        x = torch.cat([noise, class_emb], dim=1)
        
        # Project to initial feature map
        x = self.project(x)
        x = x.view(-1, self.init_channels, self.init_length)
        
        # Upsample through transposed convolution layers
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        
        # Final layer to match output channels
        x = self.final(x)
        
        return x


class Discriminator(nn.Module):
    """WGAN Discriminator with spectral normalization, ECA attention, and class conditioning
    
    AC-GAN style: Discriminator outputs both real/fake prediction and class prediction
    """
    
    def __init__(self, signal_length: int = 1024, num_classes: int = 24, channels: int = 2,
                 use_eca: bool = True, use_spectral_norm: bool = True, use_multi_scale: bool = True):
        super().__init__()
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        self.use_eca = use_eca
        self.use_spectral_norm = use_spectral_norm
        
        def maybe_sn(module):
            return spectral_norm(module) if self.use_spectral_norm else module

        # Class embedding for conditional discrimination
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        self.multi_scale = MultiScaleFeatureExtractor(channels, 64, use_multi_scale)
        
        layers = []
        
        layers.append(maybe_sn(nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)))
        layers.append(nn.InstanceNorm1d(128, affine=True))  # WGAN-GP: InstanceNorm instead of BatchNorm
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_eca:
            layers.append(ECANet(128))

        layers.append(maybe_sn(nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)))
        layers.append(nn.InstanceNorm1d(256, affine=True))  # WGAN-GP: InstanceNorm instead of BatchNorm
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_eca:
            layers.append(ECANet(256))

        layers.append(maybe_sn(nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)))
        layers.append(nn.InstanceNorm1d(512, affine=True))  # WGAN-GP: InstanceNorm instead of BatchNorm
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.features = nn.Sequential(*layers)
        
        # Real/fake prediction head
        self.head = nn.Sequential(
            maybe_sn(nn.Linear(512 + 64, 256)),  # +64 for class embedding
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Linear(256, 1))
        )
        
        # Class prediction head (AC-GAN auxiliary classifier)
        self.aux_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input signal (batch, 2, 1024)
            labels: Class labels (batch,) - optional, for conditional discrimination
        Returns:
            Real/fake prediction score (batch, 1)
        """
        x = self.multi_scale(x)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Add class condition
        if labels is not None:
            class_emb = self.class_embedding(labels)
            features_with_class = torch.cat([features, class_emb], dim=1)
        else:
            # Use zero padding when unconditional
            class_emb = torch.zeros(features.size(0), 64, device=features.device)
            features_with_class = torch.cat([features, class_emb], dim=1)
        
        return self.head(features_with_class)
    
    def forward_with_aux(self, x: torch.Tensor, labels: torch.Tensor = None):
        """Return real/fake prediction and class prediction"""
        x = self.multi_scale(x)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Real/fake prediction
        if labels is not None:
            class_emb = self.class_embedding(labels)
            features_with_class = torch.cat([features, class_emb], dim=1)
        else:
            class_emb = torch.zeros(features.size(0), 64, device=features.device)
            features_with_class = torch.cat([features, class_emb], dim=1)
        
        validity = self.head(features_with_class)
        class_pred = self.aux_classifier(features)
        
        return validity, class_pred


class Classifier(nn.Module):
    """Modulation Classifier with Dropout"""

    def __init__(self, signal_length: int = 1024, num_classes: int = 24, channels: int = 2,
                 use_eca: bool = True, use_multi_scale: bool = True, use_residual: bool = False,
                 dropout_rate: float = 0.5, hidden_scale: float = 0.5):  # Added hidden_scale parameter
        super().__init__()
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
        # Adjust hidden layer sizes based on hidden_scale
        h1, h2, h3, h4 = int(64 * hidden_scale), int(128 * hidden_scale), int(256 * hidden_scale), int(512 * hidden_scale)

        self.multi_scale = MultiScaleFeatureExtractor(channels, h1, use_multi_scale)

        self.conv2 = nn.Sequential(
            nn.Conv1d(h1, h2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2, inplace=True),
            ECANet(h2) if use_eca else nn.Identity(),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(h2, h3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.2, inplace=True),
            ECANet(h3) if use_eca else nn.Identity(),
            nn.Dropout(dropout_rate)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(h3, h4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(h4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(h4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multi_scale(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class WGANECANet(nn.Module):
    """
    WGAN-ECANet: Multi-scale Attention with Dynamic Gradient Penalty
    Unified model integrating generator, discriminator, and classifier
    """
    
    def __init__(self, noise_dim: int = 100, signal_length: int = 1024,
                 num_classes: int = 24, channels: int = 2,
                 use_eca: bool = True, use_spectral_norm: bool = True,
                 use_multi_scale: bool = True, use_residual: bool = False,
                 dropout_rate: float = 0.3, hidden_scale: float = 1.0):
        super().__init__()

        self.generator = Generator(noise_dim, signal_length, num_classes, channels)
        self.discriminator = Discriminator(signal_length, num_classes, channels,
                                          use_eca, use_spectral_norm, use_multi_scale)
        self.classifier = Classifier(signal_length, num_classes, channels,
                                    use_eca, use_multi_scale, use_residual, dropout_rate, hidden_scale)
        self.gradient_penalty = DynamicGradientPenalty()
        
    def forward(self, x: torch.Tensor, mode: str = 'classify') -> Dict[str, torch.Tensor]:
        if mode == 'classify':
            logits = self.classifier(x)
            return {'logits': logits}
        
        elif mode == 'discriminate':
            return self.discriminator(x)
        
        elif mode == 'generate':
            raise ValueError("Generate mode requires noise and labels")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def generate(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.generator(noise, labels)
    
    def get_gradient_penalty(self, discriminator, real_samples, fake_samples, device):
        return self.gradient_penalty.compute_penalty(discriminator, real_samples, fake_samples, device)
