"""Line-level MCTS search for code generation.

Each MCTS "move" generates an entire line of code. The search uses the model's
policy head to propose candidate lines and the value head to evaluate states.
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

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

    def expand_node(
        self,
        node: MCTSNode,
        prompt_ids: List[int],
    ):
        """Expand a node by generating candidate child lines.

        Each child represents a candidate next line of code.
        """
        device = self.model.device

        # Build full token sequence: prompt + code so far
        code_tokens = node.get_cumulative_tokens()
        all_tokens = prompt_ids + code_tokens
        input_ids = torch.tensor([all_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Generate candidate lines
        candidates = self.generate_candidate_lines(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=node.kv_cache,
            num_candidates=self.config.candidate_lines_k,
        )

        if not candidates:
            return

        # Compute priors from log-probabilities (softmax normalized)
        log_probs = torch.tensor([lp for _, _, lp in candidates])
        priors = F.softmax(log_probs, dim=0).tolist()

        # Create children
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

    def evaluate_node(self, node: MCTSNode, prompt_ids: List[int]) -> float:
        """Get value estimate for a node using the value head."""
        device = self.model.device
        code_tokens = node.get_cumulative_tokens()
        all_tokens = prompt_ids + code_tokens

        input_ids = torch.tensor([all_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        out = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out["value"].item()

    def run_simulation(self, root: MCTSNode, prompt_ids: List[int]):
        """Run one MCTS simulation: select → expand → evaluate → backpropagate."""
        node = root
        search_path = [node]

        # === SELECTION: traverse tree using PUCT ===
        while node.is_expanded and not node.is_terminal:
            node = select_child(node, self.config.c_puct)
            search_path.append(node)

        # === EXPANSION + EVALUATION ===
        if node.is_terminal:
            value = node.terminal_reward if node.terminal_reward is not None else 0.0
        elif node.depth >= self.config.max_lines:
            # Hit max depth — treat as terminal with value estimate
            value = self.evaluate_node(node, prompt_ids)
        else:
            # Expand this leaf
            self.expand_node(node, prompt_ids)
            # Evaluate the expanded node
            value = self.evaluate_node(node, prompt_ids)

            # Early pruning: if value too low, don't bother exploring
            if value < self.config.value_pruning_threshold and node.depth > 3:
                node.is_terminal = True
                node.terminal_reward = value

        # === BACKPROPAGATION ===
        for n in reversed(search_path):
            n.visit_count += 1
            n.total_value += value

    def search(
        self,
        prompt_ids: List[int],
        code_tokens_so_far: List[int],
        line_number: int,
        num_simulations: Optional[int] = None,
    ) -> MCTSResult:
        """Run full MCTS search at one decision point.

        Args:
            prompt_ids: tokenized problem prompt
            code_tokens_so_far: tokens of code generated so far
            line_number: current line number (for temperature scheduling)
            num_simulations: override default simulation count

        Returns:
            MCTSResult with improved policy and selected child
        """
        num_sims = num_simulations or self.config.num_simulations

        # Create root node representing current state
        root = MCTSNode(
            line_tokens=code_tokens_so_far if code_tokens_so_far else None,
            line_text="",
        )
        # Cache cumulative tokens
        root._cumulative_tokens = code_tokens_so_far.copy()

        # Expand root
        self.expand_node(root, prompt_ids)

        if not root.children:
            # No candidates generated — return empty result
            return MCTSResult(policy={}, selected_child_idx=0, root_value=0.0, total_simulations=0)

        # Add Dirichlet noise to root for exploration
        add_dirichlet_noise(root, self.config.dirichlet_alpha, self.config.dirichlet_weight)

        # Run simulations
        for _ in range(num_sims):
            self.run_simulation(root, prompt_ids)

        # Extract improved policy from visit counts
        temperature = (
            self.config.temperature_early
            if line_number < self.config.temperature_switch_line
            else self.config.temperature_late
        )
        policy = get_mcts_policy(root, temperature)

        # Sample action from policy
        selected_idx = sample_from_policy(policy)

        return MCTSResult(
            policy=policy,
            selected_child_idx=selected_idx,
            root_value=root.q_value,
            total_simulations=sum(c.visit_count for c in root.children),
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
            self.expand_node(root, prompt_ids)

            if selected_child_idx >= len(root.children):
                break

            selected = root.children[min(selected_child_idx, len(root.children) - 1)]

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
