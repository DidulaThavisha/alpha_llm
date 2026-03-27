"""MCTS utility functions: PUCT selection, policy extraction, noise injection."""

import math
import numpy as np
import torch
from typing import List, Tuple, Dict

from .node import MCTSNode


def puct_score(parent: MCTSNode, child: MCTSNode, c_puct: float) -> float:
    """Compute PUCT score for child selection (AlphaZero formula).

    score = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(child))
    """
    exploration = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.q_value + exploration


def select_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Select the child with the highest PUCT score."""
    best_score = -float("inf")
    best_child = None

    for child in node.children:
        score = puct_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def add_dirichlet_noise(node: MCTSNode, alpha: float, weight: float):
    """Add Dirichlet noise to root node's children priors for exploration.

    new_prior = (1 - weight) * prior + weight * noise
    """
    if not node.children:
        return

    noise = np.random.dirichlet([alpha] * len(node.children))
    for child, n in zip(node.children, noise):
        child.prior = (1 - weight) * child.prior + weight * n


def get_mcts_policy(node: MCTSNode, temperature: float = 1.0) -> Dict[int, float]:
    """Extract improved policy from visit counts.

    Args:
        node: root node after MCTS search
        temperature: controls exploration (0 = greedy, 1 = proportional)

    Returns:
        Dict mapping child index to probability
    """
    if not node.children:
        return {}

    visits = [c.visit_count for c in node.children]

    if temperature == 0.0:
        # Greedy: pick the most visited
        best_idx = int(np.argmax(visits))
        policy = {i: 0.0 for i in range(len(visits))}
        policy[best_idx] = 1.0
        return policy

    # Apply temperature
    adjusted = [v ** (1.0 / temperature) for v in visits]
    total = sum(adjusted)
    if total == 0:
        # Uniform if no visits
        n = len(adjusted)
        return {i: 1.0 / n for i in range(n)}

    return {i: a / total for i, a in enumerate(adjusted)}


def sample_from_policy(policy: Dict[int, float]) -> int:
    """Sample a child index from the MCTS policy distribution."""
    indices = list(policy.keys())
    probs = [policy[i] for i in indices]
    return int(np.random.choice(indices, p=probs))


def top_k_indices(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-k token indices and their log-probabilities.

    Args:
        logits: [vocab_size] raw logits
        k: number of top tokens

    Returns:
        (top_k_indices, top_k_log_probs) each of shape [k]
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    top_k_log_probs, top_k_idx = torch.topk(log_probs, k)
    return top_k_idx, top_k_log_probs
