"""Dual-headed AlphaCode model: policy (next token) + value (win probability).

Wraps a pretrained causal LM (Qwen2.5-Coder-1.5B-Instruct) with an added
value head. The original lm_head serves as the policy head.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict, Any

from .value_head import ValueHead
from config import ModelConfig


class AlphaCodeModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Load pretrained model
        dtype = getattr(torch, self.config.precision)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect hidden size from the loaded model
        hidden_size = self.backbone.config.hidden_size

        # Value head — trained from scratch, kept in float32 regardless of backbone dtype
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

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask
            past_key_values: cached key-values for incremental decoding
            use_cache: whether to return updated KV cache

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

        # Value from the last token position
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
        """Apply LoRA adapters for efficient fine-tuning."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        print(f"LoRA applied. Trainable params: {self.backbone.print_trainable_parameters()}")

    def to_device(self, device: str) -> "AlphaCodeModel":
        """Move model to device."""
        self.backbone = self.backbone.to(device)
        self.value_head = self.value_head.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def vocab_size(self) -> int:
        return self.backbone.config.vocab_size
