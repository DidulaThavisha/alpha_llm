"""Dual-headed AlphaCode model: policy (next token) + value (win probability).

Wraps a pretrained causal LM (Qwen2.5-Coder / Qwen3) via Unsloth's
FastLanguageModel for faster training and lower memory usage.
The original lm_head serves as the policy head; a bolted-on ValueHead
predicts P(win) from the last token's hidden state.
"""
from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .value_head import ValueHead
from config import ModelConfig


class AlphaCodeModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        dtype = getattr(torch, self.config.precision)
        self.backbone, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.name,
            max_seq_length=self.config.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Unsloth may return a Processor (e.g. for VL models) instead of a plain tokenizer.
        # Unwrap to the underlying tokenizer so .encode() / .decode() are available.
        if hasattr(self.tokenizer, 'tokenizer'):
            self.tokenizer = self.tokenizer.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # VL models (e.g. Qwen3.5) nest hidden_size under text_config
        cfg = self.backbone.config
        hidden_size = getattr(cfg, 'hidden_size', None) or cfg.text_config.hidden_size

        # Value head — trained from scratch, always in float32 to avoid LayerNorm overflow
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            intermediate_size=self.config.value_head_hidden,
            dropout=self.config.value_head_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass returning policy logits, value, and KV cache.

        Returns:
            dict with keys:
                logits: [batch, seq_len, vocab_size] — policy logits
                value: [batch, 1] — win probability estimate
                past_key_values: updated KV cache
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        logits = outputs.logits  # [batch, seq_len, vocab_size]
        hidden_states = outputs.hidden_states[-1]  # last layer: [batch, seq_len, hidden]

        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        value = self.value_head(last_hidden)  # [batch, 1]

        return {
            "logits": logits,
            "value": value,
            "past_key_values": outputs.past_key_values,
        }

    def get_policy_and_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, float, Optional[Tuple]]:
        """Convenience method for MCTS: returns (log_probs, value_scalar, kv_cache)."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, past_key_values)

        logits = out["logits"][:, -1, :]  # [batch, vocab] — last position
        log_probs = torch.log_softmax(logits, dim=-1)
        value_scalar = out["value"].item()

        return log_probs, value_scalar, out["past_key_values"]

    def apply_lora(self):
        """Apply LoRA adapters via Unsloth's optimized get_peft_model."""
        self.backbone = FastLanguageModel.get_peft_model(
            self.backbone,
            r=self.config.lora_rank,
            target_modules=self.config.lora_target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        self.backbone.print_trainable_parameters()

    def for_inference(self):
        """Switch backbone to Unsloth's optimized inference mode (2x faster)."""
        FastLanguageModel.for_inference(self.backbone)

    def for_training(self):
        """Switch backbone back to training mode after for_inference()."""
        FastLanguageModel.for_training(self.backbone)

    def to_device(self, device: str) -> "AlphaCodeModel":
        """Move value head to device (backbone placement is managed by Unsloth)."""
        self.value_head = self.value_head.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def vocab_size(self) -> int:
        return self.backbone.config.vocab_size
