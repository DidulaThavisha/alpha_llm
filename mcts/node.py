"""MCTS node for line-level code generation tree search."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import torch


@dataclass
class MCTSNode:
    """A node in the MCTS tree. Each node represents a line of code appended to the solution."""

    # The line of code (tokens) this node represents
    line_tokens: Optional[List[int]] = None
    line_text: str = ""

    # Tree structure
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)

    # MCTS statistics
    visit_count: int = 0       # N(s)
    total_value: float = 0.0   # W(s) — sum of backpropagated values
    prior: float = 0.0         # P(s) — policy prior from the network

    # State
    is_terminal: bool = False
    terminal_reward: Optional[float] = None

    # KV cache reference for incremental decoding
    kv_cache: Optional[Tuple] = None

    # Full token sequence from root to this node (cached for efficiency)
    _cumulative_tokens: Optional[List[int]] = None

    @property
    def q_value(self) -> float:
        """Mean action value Q(s) = W(s) / N(s)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def get_cumulative_tokens(self) -> List[int]:
        """Get full token sequence from root to this node."""
        if self._cumulative_tokens is not None:
            return self._cumulative_tokens

        tokens = []
        node = self
        path = []
        while node is not None:
            if node.line_tokens:
                path.append(node.line_tokens)
            node = node.parent

        for chunk in reversed(path):
            tokens.extend(chunk)

        self._cumulative_tokens = tokens
        return tokens

    def invalidate_cache(self):
        """Invalidate cumulative tokens cache (call when tree structure changes)."""
        self._cumulative_tokens = None
        for child in self.children:
            child.invalidate_cache()

    def best_child_by_visits(self) -> Optional["MCTSNode"]:
        """Select child with highest visit count (used for final move selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visit_count)

    def get_children_visit_distribution(self) -> Dict[str, float]:
        """Get normalized visit distribution over children (the improved policy)."""
        total = sum(c.visit_count for c in self.children)
        if total == 0:
            return {}
        return {
            c.line_text: c.visit_count / total
            for c in self.children
        }

    def free_kv_cache(self):
        """Free KV cache memory for this node and all descendants."""
        self.kv_cache = None
        for child in self.children:
            child.free_kv_cache()
