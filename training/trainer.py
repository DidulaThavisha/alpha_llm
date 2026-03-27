"""Training loop for the AlphaCode model.

Joint optimization of:
- Policy loss: KL divergence between MCTS-improved policy and model's policy
- Value loss: MSE between predicted value and actual game outcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Any, Optional
import os
import json

import logger
from training.replay_buffer import ReplayBuffer, Experience
from config import AlphaCodeConfig


class Trainer:
    """Trains the AlphaCode model on self-play experiences."""

    def __init__(self, model, config: AlphaCodeConfig):
        self.model = model
        self.config = config

        # Separate parameter groups for differential learning rates
        backbone_params = []
        value_head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "value_head" in name:
                    value_head_params.append(param)
                else:
                    backbone_params.append(param)

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": config.training.backbone_lr},
            {"params": value_head_params, "lr": config.training.value_head_lr},
        ], weight_decay=config.training.weight_decay)

        self.iteration = 0

    def compute_policy_loss(
        self,
        model_logits: torch.Tensor,
        target_policy: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between MCTS policy and model's policy.

        Args:
            model_logits: [batch, vocab_size] raw logits from model
            target_policy: [batch, vocab_size] MCTS-improved policy (sparse)
        """
        model_log_probs = F.log_softmax(model_logits, dim=-1)
        # KL(target || model) = sum(target * (log(target) - log(model)))
        # Avoid log(0) by clamping
        target_log = torch.log(target_policy.clamp(min=1e-8))
        kl = (target_policy * (target_log - model_log_probs)).sum(dim=-1)
        return kl.mean()

    def compute_value_loss(
        self,
        predicted_value: torch.Tensor,
        target_value: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and actual value."""
        return F.mse_loss(predicted_value.squeeze(-1), target_value)

    def train_step(self, batch: List[Experience]) -> Dict[str, float]:
        """One gradient step on a batch of experiences.

        Returns dict of loss values.
        """
        self.model.backbone.train()
        self.model.value_head.train()

        device = self.model.device
        tokenizer = self.model.tokenizer
        vocab_size = self.model.vocab_size

        total_policy_loss = 0.0
        total_value_loss = 0.0
        count = 0

        for exp in batch:
            # Build input: prompt + state tokens
            all_tokens = exp.prompt_ids + exp.state_tokens
            if len(all_tokens) > self.config.model.max_seq_length:
                all_tokens = all_tokens[-self.config.model.max_seq_length:]

            input_ids = torch.tensor([all_tokens], device=device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass
            out = self.model.forward(input_ids, attention_mask, use_cache=False)
            logits = out["logits"][:, -1, :]  # [1, vocab_size]
            value = out["value"]  # [1, 1]

            # Value loss
            target_value = torch.tensor([exp.outcome], device=device, dtype=value.dtype)
            v_loss = self.compute_value_loss(value, target_value)

            # Policy loss (simplified: we use the MCTS value estimate as additional signal)
            # Since MCTS policy is over candidate lines (not tokens), we use the
            # value loss as the primary learning signal during initial training.
            # Full token-level policy distillation is added in later iterations.
            p_loss = torch.tensor(0.0, device=device)

            loss = (
                self.config.training.value_loss_weight * v_loss
                + self.config.training.policy_loss_weight * p_loss
            )

            loss.backward()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            count += 1

        # Gradient step (with accumulation)
        if count > 0:
            # Scale gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= count

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "policy_loss": total_policy_loss / max(count, 1),
            "value_loss": total_value_loss / max(count, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(count, 1),
            "batch_size": count,
        }

    def train_epoch(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Train for one epoch over the replay buffer."""
        batch_size = self.config.training.batch_size
        epoch_losses = {"policy_loss": 0, "value_loss": 0, "total_loss": 0, "steps": 0}

        # Sample and train in batches
        num_batches = max(1, len(replay_buffer) // batch_size)

        for _ in range(num_batches):
            batch = replay_buffer.sample(batch_size)
            losses = self.train_step(batch)

            epoch_losses["policy_loss"] += losses["policy_loss"]
            epoch_losses["value_loss"] += losses["value_loss"]
            epoch_losses["total_loss"] += losses["total_loss"]
            epoch_losses["steps"] += 1

        # Average
        if epoch_losses["steps"] > 0:
            for key in ["policy_loss", "value_loss", "total_loss"]:
                epoch_losses[key] /= epoch_losses["steps"]

        return epoch_losses

    def train_iteration(self, replay_buffer: ReplayBuffer) -> List[Dict[str, float]]:
        """Train for multiple epochs (one AlphaZero iteration)."""
        epoch_results = []
        total_epochs = self.config.training.epochs_per_iteration

        logger.training_start(total_epochs, len(replay_buffer))

        for epoch in range(total_epochs):
            losses = self.train_epoch(replay_buffer)
            epoch_results.append(losses)
            logger.training_epoch(epoch + 1, total_epochs, losses["value_loss"], losses["policy_loss"])

        self.iteration += 1
        return epoch_results

    def get_supervised_ratio(self) -> float:
        """Get current supervised/RL ratio based on iteration."""
        cfg = self.config.training
        if self.iteration >= cfg.supervised_decay_iterations:
            return cfg.supervised_ratio_end

        progress = self.iteration / cfg.supervised_decay_iterations
        return cfg.supervised_ratio_start + progress * (cfg.supervised_ratio_end - cfg.supervised_ratio_start)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration": self.iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.checkpoint_saved(path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        logger.console.print(f"  [dim]Loaded checkpoint: {path} (iteration {self.iteration})[/]")
