import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math
from torch.nn.utils import spectral_norm


class MultiScaleECABlock(nn.Module):

    def __init__(self, channels: int, scales: list = [3, 5, 7], reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.scales = scales
        self._record = False
        self._name = None
        self._collector = None
        self.last_attention = None
        
        self.adaptive_kernel = self._get_adaptive_kernel_size(channels)
        
        self.eca_branches = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
            for k in scales
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
        self.channel_importance = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def _get_adaptive_kernel_size(self, channels: int) -> int:
        k = int(abs((math.log(channels, 2) + 1) / 2))
        k = k if k % 2 else k + 1
        return max(3, k)
    
    def enable_recording(self, name: str, collector: Dict[str, list]):
        self._record = True
        self._name = name
        self._collector = collector
        if name not in collector:
            collector[name] = []
    
    def disable_recording(self):
        self._record = False
        self._name = None
        self._collector = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()
        
        y = F.adaptive_avg_pool1d(x, 1)
        
        multi_scale_attention = []
        for i, eca_branch in enumerate(self.eca_branches):
            y_t = y.transpose(-1, -2)
            attention = eca_branch(y_t).transpose(-1, -2)
            attention = torch.sigmoid(attention)
            multi_scale_attention.append(attention * self.scale_weights[i])
        
        if len(multi_scale_attention) > 0:
            device = multi_scale_attention[0].device
            stacked_attention = torch.stack([att.to(device) for att in multi_scale_attention])
            fused_attention = torch.mean(stacked_attention, dim=0)
        else:
            fused_attention = torch.ones_like(x[:, :1, :])
        
        channel_weights = self.channel_importance(x)
        final_attention = fused_attention * channel_weights
        
        if self._record and self._collector is not None and self._name is not None:
            try:
                att = final_attention.detach().mean(dim=0).view(-1).cpu().numpy()
                self.last_attention = att
                self._collector[self._name].append(att)
            except Exception:
                pass
        
        return x * final_attention


class DynamicGradientPenalty:

    def __init__(self, initial_lambda: float = 10.0, min_lambda: float = 1.0, max_lambda: float = 50.0):
        self.initial_lambda = initial_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.current_lambda = initial_lambda
        self.gradient_history = []
        
    def compute_penalty(self, discriminator, real_samples, fake_samples, device):
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1).to(device)
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
        
        self.gradient_history.append(gradient_norm.mean().item())
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        
        self._update_lambda()
        
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty
    
    def _update_lambda(self):
        if len(self.gradient_history) < 10:
            return
            
        recent_gradients = self.gradient_history[-10:]
        gradient_std = np.std(recent_gradients)
        gradient_mean = np.mean(recent_gradients)
        
        if gradient_std > 0.5:
            self.current_lambda = min(self.current_lambda * 1.1, self.max_lambda)
        elif gradient_std < 0.1 and gradient_mean < 0.8:
            self.current_lambda = max(self.current_lambda * 0.95, self.min_lambda)


class ContrastiveLearningHead(nn.Module):

    def __init__(self, feature_dim: int, projection_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
        self.prototypes = nn.Parameter(torch.randn(24, projection_dim))
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        projected = F.normalize(self.projection(features), dim=1)
        
        prototype_sim = torch.mm(projected, F.normalize(self.prototypes, dim=1).t()) / self.temperature
        
        outputs = {
            'projected_features': projected,
            'prototype_similarities': prototype_sim
        }
        
        if labels is not None:
            contrastive_loss = self._compute_contrastive_loss(projected, labels)
            outputs['contrastive_loss'] = contrastive_loss
            
        return outputs
    
    def _compute_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        similarity_matrix = torch.mm(features, features.t()) / self.temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        mask = mask - torch.eye(batch_size, device=features.device)
        
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        positive_loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return positive_loss.mean()


class EnhancedGenerator(nn.Module):

    def __init__(self, noise_dim: int = 100, signal_length: int = 1024, num_classes: int = 24, channels: int = 2):
        super().__init__()
        self.noise_dim = noise_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        
        self.class_embedding = nn.Embedding(num_classes, 50)
        
        self.main = nn.Sequential(
            nn.Linear(noise_dim + 50, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, signal_length * channels),
            nn.Tanh()
        )
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_emb = self.class_embedding(labels)
        
        input_tensor = torch.cat([noise, class_emb], dim=1)
        
        output = self.main(input_tensor)
        output = output.view(-1, self.channels, self.signal_length)
        
        return output


class EnhancedDiscriminator(nn.Module):

    def __init__(self, signal_length: int = 1024, num_classes: int = 24, channels: int = 2, use_eca: bool = True, use_spectral_norm: bool = False, norm_type: str = 'none'):
        super().__init__()
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        self.use_spectral_norm = use_spectral_norm
        self.norm_type = norm_type
        
        def maybe_sn(module):
            return spectral_norm(module) if self.use_spectral_norm else module

        def maybe_norm(channels: int):
            if self.norm_type == 'batch':
                return nn.BatchNorm1d(channels)
            elif self.norm_type == 'instance':
                return nn.InstanceNorm1d(channels, affine=True)
            else:
                return nn.Identity()

        layers = []
        layers.append(maybe_sn(nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(maybe_sn(nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)))
        layers.append(maybe_norm(128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(MultiScaleECABlock(128) if use_eca else nn.Identity())

        layers.append(maybe_sn(nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)))
        layers.append(maybe_norm(256))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(MultiScaleECABlock(256) if use_eca else nn.Identity())

        layers.append(maybe_sn(nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)))
        layers.append(maybe_norm(512))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.AdaptiveAvgPool1d(1))

        self.features = nn.Sequential(*layers)
        
        head_layers = [
            maybe_sn(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Linear(256, 1))
        ]
        self.discriminator_head = nn.Sequential(*head_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        output = self.discriminator_head(features)
        return output


class TransformerAttentionBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x

class EnhancedClassifier(nn.Module):

    def __init__(self, signal_length: int = 1024, num_classes: int = 24, channels: int = 2, 
                 use_eca: bool = True, use_transformer_attention: bool = False, 
                 num_attention_heads: int = 8, attention_dropout: float = 0.1,
                 use_residual_connections: bool = False, feature_pyramid_levels: int = 3):
        super().__init__()
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        self.use_transformer_attention = use_transformer_attention
        self.use_residual_connections = use_residual_connections
        self.feature_pyramid_levels = feature_pyramid_levels
        
        self.feature_layers = nn.ModuleList()
        
        layers1 = [
            nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        layers2 = [
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleECABlock(128) if use_eca else nn.Identity()
        ]
        
        layers3 = [
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleECABlock(256) if use_eca else nn.Identity()
        ]
        
        layers4 = [
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        ]
        
        self.feature_layers.append(nn.Sequential(*layers1))
        self.feature_layers.append(nn.Sequential(*layers2))
        self.feature_layers.append(nn.Sequential(*layers3))
        self.feature_layers.append(nn.Sequential(*layers4))
        
        if use_residual_connections:
            self.residual_conv1 = nn.Conv1d(64, 128, kernel_size=1)
            self.residual_conv2 = nn.Conv1d(128, 256, kernel_size=1)
            self.residual_conv3 = nn.Conv1d(256, 512, kernel_size=1)
        else:
            self.residual_conv1 = None
            self.residual_conv2 = None
            self.residual_conv3 = None
        
        if use_transformer_attention:
            self.transformer_attention = TransformerAttentionBlock(
                512, num_attention_heads, attention_dropout
            )
            actual_pyramid_levels = min(feature_pyramid_levels, 4)
            self.feature_projection = nn.Linear(512 * actual_pyramid_levels, 512)
        else:
            self.transformer_attention = None
            self.feature_projection = None
        
        if feature_pyramid_levels > 1:
            fusion_layers = min(feature_pyramid_levels - 1, 3)
            self.pyramid_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(64, 512, kernel_size=1),
                    nn.AdaptiveAvgPool1d(1)
                ) if i == 0 else
                nn.Sequential(
                    nn.Conv1d(128, 512, kernel_size=1),
                    nn.AdaptiveAvgPool1d(1)
                ) if i == 1 else
                nn.Sequential(
                    nn.Conv1d(256, 512, kernel_size=1),
                    nn.AdaptiveAvgPool1d(1)
                ) for i in range(fusion_layers)
            ])
        else:
            self.pyramid_fusion = nn.ModuleList()
        
        self.pyramid_conv = nn.ModuleList([
            nn.Conv1d(64, 512, kernel_size=1),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Conv1d(256, 512, kernel_size=1)
        ])
        
        classifier_input_dim = 512 if use_transformer_attention else 512 * feature_pyramid_levels
        self.classifier_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        self.contrastive_head = ContrastiveLearningHead(classifier_input_dim)
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_features: bool = False) -> Dict[str, torch.Tensor]:
        pyramid_features = []
        
        feat1 = self.feature_layers[0](x)
        
        if self.feature_pyramid_levels >= 2:
            if len(self.pyramid_fusion) > 0:
                pyramid_features.append(self.pyramid_fusion[0](feat1))
            else:
                pyramid_features.append(F.adaptive_avg_pool1d(self.pyramid_conv[0](feat1), 1))
        
        if self.use_residual_connections:
            residual = self.residual_conv1(feat1)
            residual = F.avg_pool1d(residual, kernel_size=2, stride=2)
            feat2 = self.feature_layers[1](feat1) + residual
        else:
            feat2 = self.feature_layers[1](feat1)
            
        if self.feature_pyramid_levels >= 3:
            if len(self.pyramid_fusion) > 1:
                pyramid_features.append(self.pyramid_fusion[1](feat2))
            else:
                pyramid_features.append(F.adaptive_avg_pool1d(self.pyramid_conv[1](feat2), 1))
        
        if self.use_residual_connections:
            residual = self.residual_conv2(feat2)
            residual = F.avg_pool1d(residual, kernel_size=2, stride=2)
            feat3 = self.feature_layers[2](feat2) + residual
        else:
            feat3 = self.feature_layers[2](feat2)
            
        if self.feature_pyramid_levels >= 4:
            if len(self.pyramid_fusion) > 2:
                pyramid_features.append(self.pyramid_fusion[2](feat3))
            else:
                pyramid_features.append(F.adaptive_avg_pool1d(self.pyramid_conv[2](feat3), 1))
        
        if self.use_residual_connections:
            residual = self.residual_conv3(feat3)
            residual = F.adaptive_avg_pool1d(residual, 1)
            final_feat = self.feature_layers[3](feat3) + residual
        else:
            final_feat = self.feature_layers[3](feat3)
            
        pyramid_features.append(final_feat)
        
        if self.feature_pyramid_levels > 1:
            fused_features = torch.cat(pyramid_features, dim=1)
            features = fused_features.view(fused_features.size(0), -1)
        else:
            features = final_feat.view(final_feat.size(0), -1)
        
        if self.use_transformer_attention:
            if features.size(1) != 512:
                features = self.feature_projection(features)
            
            seq_features = features.unsqueeze(1)
            attended_features = self.transformer_attention(seq_features)
            features = attended_features.squeeze(1)
        
        logits = self.classifier_head(features)
        
        outputs = {'logits': logits}
        
        if return_features:
            outputs['features'] = features
            
        if labels is not None:
            contrastive_outputs = self.contrastive_head(features, labels)
            outputs.update(contrastive_outputs)
            
        return outputs


class EnhancedWGANECANet(nn.Module):

    def __init__(self, 
                 noise_dim: int = 100, 
                 signal_length: int = 1024, 
                 num_classes: int = 24, 
                 channels: int = 2,
                 use_eca: bool = True,
                 use_spectral_norm: bool = False,
                 discriminator_norm: str = 'none',
                 use_transformer_attention: bool = False,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 use_residual_connections: bool = False,
                 feature_pyramid_levels: int = 3):
        super().__init__()
        self.noise_dim = noise_dim
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        
        self.generator = EnhancedGenerator(noise_dim, signal_length, num_classes, channels)
        self.discriminator = EnhancedDiscriminator(signal_length, num_classes, channels, use_eca, use_spectral_norm, discriminator_norm)
        self.classifier = EnhancedClassifier(
            signal_length, num_classes, channels, use_eca,
            use_transformer_attention=use_transformer_attention,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_residual_connections=use_residual_connections,
            feature_pyramid_levels=feature_pyramid_levels
        )
        
        self.gradient_penalty = DynamicGradientPenalty()
        
        self.training_stage = 'warmup'
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, mode: str = 'classify') -> Dict[str, torch.Tensor]:
        if mode == 'classify':
            return self.classifier(x, labels, return_features=True)
        elif mode == 'discriminate':
            return {'discriminator_output': self.discriminator(x)}
        elif mode == 'generate':
            if labels is None:
                labels = torch.randint(0, self.num_classes, (x.size(0),), device=x.device)
            noise = torch.randn(x.size(0), self.noise_dim, device=x.device)
            return {'generated_signals': self.generator(noise, labels)}
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute_losses(self, real_signals: torch.Tensor, real_labels: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        batch_size = real_signals.size(0)
        losses = {}
        
        classifier_outputs = self.classifier(real_signals, real_labels, return_features=True)
        classification_loss = F.cross_entropy(classifier_outputs['logits'], real_labels)
        losses['classification_loss'] = classification_loss
        
        if 'contrastive_loss' in classifier_outputs:
            losses['contrastive_loss'] = classifier_outputs['contrastive_loss']
        
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        fake_signals = self.generator(noise, fake_labels)
        
        fake_discriminator_output = self.discriminator(fake_signals)
        generator_loss = -fake_discriminator_output.mean()
        losses['generator_loss'] = generator_loss
        
        real_discriminator_output = self.discriminator(real_signals)
        fake_discriminator_output_detached = self.discriminator(fake_signals.detach())
        
        discriminator_loss = fake_discriminator_output_detached.mean() - real_discriminator_output.mean()
        
        gradient_penalty = self.gradient_penalty.compute_penalty(
            self.discriminator, real_signals, fake_signals.detach(), device
        )
        
        discriminator_loss += gradient_penalty
        losses['discriminator_loss'] = discriminator_loss
        losses['gradient_penalty'] = gradient_penalty
        
        return losses
    
    def set_training_stage(self, stage: str):
        self.training_stage = stage
        print(f"Training stage set to: {stage}")


class EmbeddedWGANECANet(nn.Module):

    def __init__(self, signal_length: int = 1024, num_classes: int = 24, channels: int = 2):
        super().__init__()
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.channels = channels
        
        self.features = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, stride=2, padding=3, groups=channels),
            nn.Conv1d(channels, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
            
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, groups=32),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU6(inplace=True),
            
            MultiScaleECABlock(64, scales=[3, 5], reduction=8),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU6(inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def get_model_size(self) -> Dict[str, float]:
        param_count = sum(p.numel() for p in self.parameters())
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'parameters': param_count,
            'model_size_mb': (param_size + buffer_size) / (1024 * 1024),
            'param_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024)
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EnhancedWGANECANet()
    model.to(device)
    
    batch_size = 4
    test_signals = torch.randn(batch_size, 2, 1024).to(device)
    test_labels = torch.randint(0, 24, (batch_size,)).to(device)
    
    with torch.no_grad():
        outputs = model(test_signals, test_labels, mode='classify')
        print("Classification output shape:", outputs['logits'].shape)
        print("Features shape:", outputs['features'].shape)
    
    embedded_model = EmbeddedWGANECANet()
    embedded_model.to(device)
    
    with torch.no_grad():
        embedded_output = embedded_model(test_signals)
        print("Embedded model output shape:", embedded_output.shape)
        
    print("\nModel size comparison:")
    print("Enhanced model parameters:", sum(p.numel() for p in model.parameters()))
    print("Embedded model parameters:", sum(p.numel() for p in embedded_model.parameters()))
    print("Embedded model details:", embedded_model.get_model_size())
