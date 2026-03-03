#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WGAN-GP Trainer

Implements complete Wasserstein GAN with Gradient Penalty training workflow:
- Discriminator (Critic) training: maximize W(x_real) - W(x_fake)
- Generator training: minimize -W(x_fake)
- Gradient penalty: λ(‖∇_x̂ D(x̂)‖₂ - 1)²
- Dynamic gradient penalty coefficient adjustment
- Classifier joint training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import gc


class WGANTrainer:
    """
    WGAN-GP Trainer
    
    Training workflow:
    1. Train discriminator D (n_critic times)
       - Maximize: E[D(x_real)] - E[D(x_fake)]
       - Add gradient penalty
    2. Train generator G (1 time)
       - Minimize: -E[D(x_fake)]
    3. Train classifier C
       - Standard cross-entropy loss
    """
    
    def __init__(
        self,
        model,
        device,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        lr_c: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 3e-4,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        use_dynamic_gp: bool = True,
        gp_base: float = 10.0,
        gp_min: float = 5.0,
        classifier_weight: float = 1.0,
        gan_weight: float = 0.1,
        gradient_accumulation_steps: int = 2
    ):
        """
        Args:
            model: WGANECANet model
            device: Computing device
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            lr_c: Classifier learning rate
            beta1, beta2: AdamW optimizer parameters (paper: 0.9, 0.999)
            weight_decay: L2 regularization coefficient (paper: 3e-4)
            lambda_gp: Gradient penalty coefficient
            n_critic: Number of discriminator training steps per generator step
            use_dynamic_gp: Whether to use dynamic gradient penalty
            gp_base: Gradient penalty coefficient initial value
            gp_min: Gradient penalty coefficient minimum value
            classifier_weight: Classification loss weight
            gan_weight: GAN loss weight (for classifier)
            gradient_accumulation_steps: Gradient accumulation steps (paper: 2)
        """
        self.model = model
        self.device = device
        self.n_critic = n_critic
        self.classifier_weight = classifier_weight
        self.gan_weight = gan_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.lambda_gp = lambda_gp
        self.use_dynamic_gp = use_dynamic_gp
        self.gp_base = gp_base
        self.gp_min = gp_min
        self.current_lambda = gp_base
        
        self.total_steps = 0
        self.current_step = 0
        
        self.optimizer_G = optim.AdamW(
            model.generator.parameters(),
            lr=lr_g, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay
        )
        self.optimizer_D = optim.AdamW(
            model.discriminator.parameters(),
            lr=lr_d, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay
        )
        self.optimizer_C = optim.AdamW(
            model.classifier.parameters(),
            lr=lr_c, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay
        )
        
        # Use ReduceLROnPlateau learning rate scheduler (paper setting)
        # Monitor validation accuracy, multiply learning rate by 0.5 after 5 consecutive epochs without improvement
        self.schedulers = {
            'G': optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G, mode='max', factor=0.5, patience=5
            ),
            'D': optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D, mode='max', factor=0.5, patience=5
            ),
            'C': optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_C, mode='max', factor=0.5, patience=5
            )
        }
        
        self.criterion_classifier = nn.CrossEntropyLoss()
        
    def set_total_steps(self, total_steps: int):
        """Set total training steps for dynamic gradient penalty"""
        self.total_steps = total_steps
        self.model.gradient_penalty.set_total_steps(total_steps)
        
    def _update_lambda_gp(self):
        """Dynamically update gradient penalty coefficient"""
        if not self.use_dynamic_gp:
            return
            
        t = self.current_step
        T = self.total_steps
        
        if T == 0:
            return
            
        if t < 0.3 * T:
            self.current_lambda = self.gp_base
        elif t < 0.7 * T:
            progress = (t - 0.3 * T) / (0.4 * T)
            self.current_lambda = self.gp_base - (self.gp_base - self.gp_min) * progress
        else:
            self.current_lambda = self.gp_min
            
        self.model.gradient_penalty.current_lambda = self.current_lambda
        
    def compute_gradient_penalty(
        self, 
        real_samples: torch.Tensor, 
        fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty
        
        GP = λ(‖∇_x̂ D(x̂)‖₂ - 1)²
        
        where x̂ = εx_real + (1-ε)x_fake
        """
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        d_interpolates = self.model.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(
        self,
        real_samples: torch.Tensor,
        labels: torch.Tensor,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """
        Train discriminator

        Loss function:
        L_D = -E[D(x_real)] + E[D(x_fake)] + λ * GP
        """
        # Only zero gradients at first step
        if accumulation_step == 0:
            self.optimizer_D.zero_grad()

        d_real = self.model.discriminator(real_samples)
        loss_d_real = -d_real.mean()

        batch_size = real_samples.size(0)
        noise = torch.randn(batch_size, self.model.generator.noise_dim, device=self.device)
        fake_samples = self.model.generator(noise, labels)

        d_fake = self.model.discriminator(fake_samples.detach())
        loss_d_fake = d_fake.mean()

        gradient_penalty = self.compute_gradient_penalty(real_samples, fake_samples.detach())

        loss_D = loss_d_real + loss_d_fake + self.current_lambda * gradient_penalty

        # Gradient accumulation: divide by accumulation steps
        loss_D = loss_D / self.gradient_accumulation_steps
        loss_D.backward()

        # Only update parameters at last step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
            self.optimizer_D.step()

        return {
            'loss_D': loss_D.item() * self.gradient_accumulation_steps,
            'loss_D_real': loss_d_real.item(),
            'loss_D_fake': loss_d_fake.item(),
            'gradient_penalty': gradient_penalty.item(),
            'wasserstein_distance': (-loss_d_real - loss_d_fake).item()
        }
    
    def train_generator(
        self,
        labels: torch.Tensor,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """
        Train generator

        Loss function:
        L_G = -E[D(x_fake)]
        """
        # Only zero gradients at first step
        if accumulation_step == 0:
            self.optimizer_G.zero_grad()

        batch_size = labels.size(0)
        noise = torch.randn(batch_size, self.model.generator.noise_dim, device=self.device)
        fake_samples = self.model.generator(noise, labels)

        d_fake = self.model.discriminator(fake_samples)
        loss_G = -d_fake.mean()

        # Gradient accumulation: divide by accumulation steps
        loss_G = loss_G / self.gradient_accumulation_steps
        loss_G.backward()

        # Only update parameters at last step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)
            self.optimizer_G.step()

        return {
            'loss_G': loss_G.item() * self.gradient_accumulation_steps
        }
    
    def train_classifier(
        self,
        real_samples: torch.Tensor,
        labels: torch.Tensor,
        use_gan_loss: bool = True,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """
        Train classifier

        Loss function:
        L_C = L_classification + α * L_GAN
        """
        # Only zero gradients at first step
        if accumulation_step == 0:
            self.optimizer_C.zero_grad()

        outputs = self.model.classifier(real_samples)
        loss_classification = self.criterion_classifier(outputs, labels)

        if use_gan_loss and self.gan_weight > 0:
            with torch.no_grad():
                batch_size = labels.size(0)
                noise = torch.randn(batch_size, self.model.generator.noise_dim, device=self.device)
                fake_samples = self.model.generator(noise, labels)

            fake_outputs = self.model.classifier(fake_samples.detach())
            loss_gan = self.criterion_classifier(fake_outputs, labels)

            loss_C = self.classifier_weight * loss_classification + self.gan_weight * loss_gan
        else:
            loss_C = loss_classification
            loss_gan = torch.tensor(0.0)

        # Gradient accumulation: divide by accumulation steps
        loss_C = loss_C / self.gradient_accumulation_steps
        loss_C.backward()

        # Only update parameters at last step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)
            self.optimizer_C.step()

        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        return {
            'loss_C': loss_C.item() * self.gradient_accumulation_steps,
            'loss_classification': loss_classification.item(),
            'loss_gan': loss_gan.item() if isinstance(loss_gan, torch.Tensor) else 0.0,
            'accuracy': accuracy
        }
    
    def train_step(
        self,
        real_samples: torch.Tensor,
        labels: torch.Tensor,
        train_discriminator: bool = True,
        train_generator: bool = True,
        train_classifier: bool = True,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """
        Complete training step (supports gradient accumulation)

        Args:
            real_samples: Real signal samples
            labels: Class labels
            train_discriminator: Whether to train discriminator
            train_generator: Whether to train generator
            train_classifier: Whether to train classifier
            accumulation_step: Current gradient accumulation step
        """
        metrics = {}

        if train_discriminator:
            for _ in range(self.n_critic):
                d_metrics = self.train_discriminator(real_samples, labels, accumulation_step)
            metrics.update(d_metrics)

        if train_generator:
            g_metrics = self.train_generator(labels, accumulation_step)
            metrics.update(g_metrics)

        if train_classifier:
            c_metrics = self.train_classifier(real_samples, labels, accumulation_step=accumulation_step)
            metrics.update(c_metrics)

        # Only update global step at last step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            self.current_step += 1
            self._update_lambda_gp()

        return metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        train_discriminator: bool = True,
        train_generator: bool = True,
        train_classifier: bool = True
    ) -> Dict[str, float]:
        """
        Train one epoch (supports gradient accumulation)

        Args:
            train_loader: Training data loader
            epoch: Current epoch
            train_discriminator: Whether to train discriminator
            train_generator: Whether to train generator
            train_classifier: Whether to train classifier
        """
        self.model.train()

        epoch_metrics = {
            'loss_D': 0.0,
            'loss_G': 0.0,
            'loss_C': 0.0,
            'accuracy': 0.0,
            'wasserstein_distance': 0.0,
            'gradient_penalty': 0.0
        }
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, batch_data in enumerate(pbar):
            real_samples = batch_data['signals'].to(self.device)
            labels = batch_data['labels'].to(self.device)

            # Calculate current gradient accumulation step
            accumulation_step = batch_idx % self.gradient_accumulation_steps

            metrics = self.train_step(
                real_samples, labels,
                train_discriminator=train_discriminator,
                train_generator=train_generator,
                train_classifier=train_classifier,
                accumulation_step=accumulation_step
            )

            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            num_batches += 1

            pbar.set_postfix({
                'loss_D': f'{metrics.get("loss_D", 0):.4f}',
                'loss_G': f'{metrics.get("loss_G", 0):.4f}',
                'loss_C': f'{metrics.get("loss_C", 0):.4f}',
                'acc': f'{metrics.get("accuracy", 0)*100:.2f}%',
                'λ_gp': f'{self.current_lambda:.1f}',
                'accum': f'{accumulation_step+1}/{self.gradient_accumulation_steps}'
            })
        
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Note: ReduceLROnPlateau needs to be called after validation with validation metric
        # Not called here, will be called externally after validation
        
        return epoch_metrics
    
    def step_schedulers(self, metric: float):
        """
        Update learning rate schedulers
        
        Args:
            metric: Validation metric (e.g., validation accuracy)
        """
        for scheduler in self.schedulers.values():
            scheduler.step(metric)
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate classifier performance"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]  ')
            for batch_data in pbar:
                signals = batch_data['signals'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                
                outputs = self.model.classifier(signals)
                loss = self.criterion_classifier(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100 * correct / total
        }
    
    def generate_samples(
        self,
        num_samples: int,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples
        
        Args:
            num_samples: Number of samples to generate
            labels: Specified class labels (optional)
        """
        self.model.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.model.generator.noise_dim, device=self.device)
            
            if labels is None:
                labels = torch.randint(0, 24, (num_samples,), device=self.device)
            
            fake_samples = self.model.generator(noise, labels)
        
        return fake_samples, labels
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'classifier_state_dict': self.model.classifier.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_C_state_dict': self.optimizer_C.state_dict(),
            'metrics': metrics,
            'current_lambda': self.current_lambda,
            'current_step': self.current_step
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        
        self.current_lambda = checkpoint.get('current_lambda', self.gp_base)
        self.current_step = checkpoint.get('current_step', 0)
        
        return checkpoint['epoch'], checkpoint.get('metrics', {})
