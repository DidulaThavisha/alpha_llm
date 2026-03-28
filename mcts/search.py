"""Line-level MCTS search for code generation.

Each MCTS "move" generates an entire line of code. The search uses the model's
policy head to propose candidate lines and the value head to evaluate states.
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import logger
from .node import MCTSNode
from .utils import select_child, add_dirichlet_noise, get_mcts_policy, sample_from_policy
from .kv_cache_pool import KVCachePool, clone_kv_cache
from config import MCTSConfig


@dataclass
class MCTSResult:
    """Result of running MCTS at one decision point."""
    policy: Dict[int, float]          # improved policy over children
    selected_child_idx: int           # which child was selected
    root_value: float                 # value estimate at root
    total_simulations: int


class MCTSSearch:
    """Line-level Monte Carlo Tree Search for code generation."""

    def __init__(self, model, config: Optional[MCTSConfig] = None):
        """
        Args:
            model: AlphaCodeModel with get_policy_and_value method
            config: MCTS hyperparameters
        """
        self.model = model
        self.config = config or MCTSConfig()
        self.cache_pool = KVCachePool()

    @torch.no_grad()
    def generate_candidate_lines(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        num_candidates: int = 8,
        max_line_tokens: int = 128,
    ) -> List[Tuple[List[int], str, float]]:
        """Generate candidate lines via temperature sampling.

        Computes the base KV cache ONCE, then generates each candidate
        autoregressively from the cached prefix (no deep clone needed).

        Returns list of (token_ids, text, log_probability) for each candidate line.
        """
        device = self.model.device
        tokenizer = self.model.tokenizer
        eos_id = tokenizer.eos_token_id
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_ids[-1] if newline_ids else None

        # === Step 1: compute base KV cache from the full prefix ONCE ===
        if past_key_values is None:
            base_out = self.model.forward(input_ids, use_cache=True)
            base_kv = base_out["past_key_values"]
            base_logits = base_out["logits"][:, -1, :].float()
        else:
            base_kv = past_key_values
            base_out = self.model.forward(input_ids[:, -1:], past_key_values=base_kv, use_cache=True)
            base_kv = base_out["past_key_values"]
            base_logits = base_out["logits"][:, -1, :].float()

        candidates = []
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 0.4, 0.8]
        temperatures = temperatures[:num_candidates]

        for temp in temperatures:
            tokens = []
            log_prob_sum = 0.0

            # Sample first token from base logits (shared across all candidates)
            top_k = self.config.top_k_tokens
            top_values, top_indices = torch.topk(base_logits, top_k, dim=-1)
            scaled = top_values / max(temp, 0.1)
            probs = F.softmax(scaled, dim=-1)
            if probs.isnan().any() or (probs < 0).any():
                probs = torch.ones_like(probs) / probs.shape[-1]

            idx = torch.multinomial(probs, 1)
            token_id = top_indices[0, idx[0, 0]].item()
            log_prob_sum += F.log_softmax(scaled, dim=-1)[0, idx[0, 0]].item()
            tokens.append(token_id)

            if token_id == eos_id or token_id == newline_id:
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                candidates.append((tokens, text, log_prob_sum))
                continue

            # Continue generating from the diverged token — build a fresh KV cache
            # by feeding the base prefix + this token
            cur_input = torch.tensor([[token_id]], device=device)
            cur_out = self.model.forward(cur_input, past_key_values=base_kv, use_cache=True)
            cur_kv = cur_out["past_key_values"]

            for step in range(1, max_line_tokens):
                logits = cur_out["logits"][:, -1, :].float()
                top_values, top_indices = torch.topk(logits, top_k, dim=-1)
                scaled = top_values / max(temp, 0.1)
                probs = F.softmax(scaled, dim=-1)
                if probs.isnan().any() or (probs < 0).any():
                    probs = torch.ones_like(probs) / probs.shape[-1]

                idx = torch.multinomial(probs, 1)
                token_id = top_indices[0, idx[0, 0]].item()
                log_prob_sum += F.log_softmax(scaled, dim=-1)[0, idx[0, 0]].item()
                tokens.append(token_id)

                if token_id == eos_id or token_id == newline_id:
                    break

                cur_input = torch.tensor([[token_id]], device=device)
                cur_out = self.model.forward(cur_input, past_key_values=cur_kv, use_cache=True)
                cur_kv = cur_out["past_key_values"]

            text = tokenizer.decode(tokens, skip_special_tokens=False)
            candidates.append((tokens, text, log_prob_sum))

        # Deduplicate by text
        seen = set()
        unique = []
        for tokens, text, lp in candidates:
            if text not in seen:
                seen.add(text)
                unique.append((tokens, text, lp))

        return unique

    def _get_state_kv(self, prompt_ids: List[int], code_tokens: List[int]) -> Tuple:
        """Compute KV cache for prompt + code_tokens in one forward pass.

        Returns (kv_cache, value_estimate).
        """
        device = self.model.device
        all_tokens = prompt_ids + code_tokens
        input_ids = torch.tensor([all_tokens], device=device)
        out = self.model.forward(input_ids, use_cache=True)
        return out["past_key_values"], out["value"].item()

    def expand_and_evaluate(
        self,
        node: MCTSNode,
        prompt_ids: List[int],
    ) -> float:
        """Expand a node AND get its value in a single forward pass.

        Generates candidate child lines using the state KV cache,
        then returns the value estimate. This avoids the double forward pass
        that expand_node + evaluate_node would require.
        """
        device = self.model.device
        code_tokens = node.get_cumulative_tokens()
        all_tokens = prompt_ids + code_tokens
        input_ids = torch.tensor([all_tokens], device=device)

        # Forward pass to get both value AND base KV for candidate generation
        out = self.model.forward(input_ids, use_cache=True)
        value = out["value"].item()
        base_kv = out["past_key_values"]

        # Generate candidates reusing the computed KV cache
        candidates = self.generate_candidate_lines(
            input_ids=input_ids,
            past_key_values=base_kv,
            num_candidates=self.config.candidate_lines_k,
        )

        if not candidates:
            return value

        # Compute priors from log-probabilities
        log_probs = torch.tensor([lp for _, _, lp in candidates])
        priors = F.softmax(log_probs, dim=0).tolist()

        for (tokens, text, _), prior in zip(candidates, priors):
            is_eos = (
                len(tokens) > 0
                and tokens[-1] == self.model.tokenizer.eos_token_id
            )
            child = MCTSNode(
                line_tokens=tokens,
                line_text=text,
                parent=node,
                prior=prior,
                is_terminal=is_eos,
            )
            node.children.append(child)

        return value

    def _score_children_with_value_head(self, root: MCTSNode, prompt_ids: List[int]):
        """Score each child with the value head (1-ply lookahead).

        Instead of deep MCTS tree search (too slow for 1.5B model on MPS),
        we do a shallow but informed search: generate K candidate lines,
        then evaluate each candidate by running the value head on
        prompt + code_so_far + candidate_line.
        """
        device = self.model.device

        for child in root.children:
            if child.is_terminal:
                child.total_value = 0.5  # neutral for terminal
                child.visit_count = 1
                continue

            # Build state after appending this child's line
            child_tokens = root.get_cumulative_tokens() + (child.line_tokens or [])
            all_tokens = prompt_ids + child_tokens

            # Truncate if too long
            if len(all_tokens) > self.config.max_tokens:
                all_tokens = all_tokens[-self.config.max_tokens:]

            input_ids = torch.tensor([all_tokens], device=device)
            out = self.model.forward(input_ids, use_cache=False)
            value = out["value"].item()

            child.total_value = value
            child.visit_count = 1

    def search(
        self,
        prompt_ids: List[int],
        code_tokens_so_far: List[int],
        line_number: int,
        num_simulations: Optional[int] = None,
    ) -> MCTSResult:
        """1-ply MCTS: generate candidates, score each with value head, pick best.

        This is the tractable version for a 1.5B model on MPS:
        - 1 forward pass to generate K candidate lines
        - K forward passes to score each candidate with the value head
        - Select based on prior × value (PUCT-like)
        Total: K+1 forward passes per line decision (not K×N).
        """
        root = MCTSNode(
            line_tokens=code_tokens_so_far if code_tokens_so_far else None,
            line_text="",
        )
        root._cumulative_tokens = code_tokens_so_far.copy()

        # Expand root: 1 forward pass + K candidate line generations
        root_value = self.expand_and_evaluate(root, prompt_ids)
        root.visit_count = 1
        root.total_value = root_value

        if not root.children:
            return MCTSResult(policy={}, selected_child_idx=0, root_value=0.0, total_simulations=0)

        add_dirichlet_noise(root, self.config.dirichlet_alpha, self.config.dirichlet_weight)

        # Score each child with value head (K forward passes)
        self._score_children_with_value_head(root, prompt_ids)

        # Combine prior and value to get policy
        # score = prior * (1 - alpha) + value * alpha  (blend prior with value)
        alpha = 0.6  # weight toward value head
        scores = []
        for child in root.children:
            blended = (1 - alpha) * child.prior + alpha * child.q_value
            scores.append(blended)

        # Normalize to policy
        total = sum(scores)
        if total > 0:
            policy = {i: s / total for i, s in enumerate(scores)}
        else:
            policy = {i: 1.0 / len(scores) for i in range(len(scores))}

        temperature = (
            self.config.temperature_early
            if line_number < self.config.temperature_switch_line
            else self.config.temperature_late
        )

        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            import numpy as np
            adjusted = {k: v ** (1.0 / temperature) for k, v in policy.items()}
            total = sum(adjusted.values())
            policy = {k: v / total for k, v in adjusted.items()}

        selected_idx = sample_from_policy(policy)

        return MCTSResult(
            policy=policy,
            selected_child_idx=selected_idx,
            root_value=root_value,
            total_simulations=len(root.children),
        )

    def generate_solution(
        self,
        prompt_ids: List[int],
        use_mcts: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a complete code solution.

        If use_mcts=True: MCTS-guided line-by-line generation with trajectory.
        If use_mcts=False: fast greedy autoregressive decoding with KV cache.
        """
        if not use_mcts:
            return self._greedy_generate(prompt_ids), []

        return self._mcts_generate(prompt_ids)

    def _greedy_generate(self, prompt_ids: List[int]) -> str:
        """Fast greedy generation using KV cache — no MCTS overhead."""
        device = self.model.device
        tokenizer = self.model.tokenizer
        eos_id = tokenizer.eos_token_id

        input_ids = torch.tensor([prompt_ids], device=device)
        out = self.model.forward(input_ids, use_cache=True)
        kv = out["past_key_values"]

        generated = []
        for _ in range(self.config.max_tokens):
            logits = out["logits"][:, -1, :].float()
            token_id = logits.argmax(dim=-1).item()
            generated.append(token_id)

            if token_id == eos_id:
                break

            next_input = torch.tensor([[token_id]], device=device)
            out = self.model.forward(next_input, past_key_values=kv, use_cache=True)
            kv = out["past_key_values"]

        return tokenizer.decode(generated, skip_special_tokens=True)

    def _mcts_generate(self, prompt_ids: List[int]) -> Tuple[str, List[Dict[str, Any]]]:
        """MCTS-guided line-by-line generation with trajectory recording."""
        code_tokens = []
        trajectory = []
        tokenizer = self.model.tokenizer

        for line_num in range(self.config.max_lines):
            result = self.search(
                prompt_ids=prompt_ids,
                code_tokens_so_far=code_tokens,
                line_number=line_num,
            )

            if not result.policy:
                break

            selected_child_idx = result.selected_child_idx

            # Regenerate root to get the selected child's tokens
            root = MCTSNode(line_tokens=code_tokens if code_tokens else None)
            root._cumulative_tokens = code_tokens.copy()
            self.expand_and_evaluate(root, prompt_ids)

            if selected_child_idx >= len(root.children):
                break

            selected = root.children[min(selected_child_idx, len(root.children) - 1)]

            # Log MCTS decision
            logger.mcts_search_step(
                line_num=line_num,
                num_children=len(root.children),
                best_value=result.root_value,
                simulations=result.total_simulations,
                selected_line=selected.line_text,
            )

            trajectory.append({
                "state_tokens": code_tokens.copy(),
                "mcts_policy": result.policy,
                "selected_line": selected.line_text,
                "value_estimate": result.root_value,
            })

            code_tokens.extend(selected.line_tokens)

            if selected.is_terminal:
                break
            if len(code_tokens) >= self.config.max_tokens:
                break

        generated_code = tokenizer.decode(code_tokens, skip_special_tokens=True)
        return generated_code, trajectory
