"""Experience replay buffer for storing self-play trajectories.

Stores (state, mcts_policy, outcome) tuples from self-play games,
with sampling for training batches.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Experience:
    """Single training example from a self-play trajectory step."""
    prompt_ids: List[int]         # tokenized problem prompt
    state_tokens: List[int]       # code tokens at this decision point
    mcts_policy: Dict[int, float] # improved policy from MCTS (child_idx → prob)
    line_candidates: List[str]    # text of candidate lines (parallel to policy keys)
    outcome: float                # final game outcome: +1 win, -1 loss, partial
    value_estimate: float         # MCTS value estimate at this state
    problem_rating: int           # difficulty rating of the problem


class ReplayBuffer:
    """Rolling replay buffer for self-play experiences."""

    def __init__(self, max_size: int = 10_000):
        self.buffer: deque[Experience] = deque(maxlen=max_size)

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def add_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        prompt_ids: List[int],
        outcome: float,
        problem_rating: int,
    ):
        """Add all steps from a self-play trajectory."""
        for step in trajectory:
            exp = Experience(
                prompt_ids=prompt_ids,
                state_tokens=step["state_tokens"],
                mcts_policy=step["mcts_policy"],
                line_candidates=step.get("line_candidates", []),
                outcome=outcome,
                value_estimate=step.get("value_estimate", 0.0),
                problem_rating=problem_rating,
            )
            self.add(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences."""
        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the buffer."""
        if not self.buffer:
            return {"size": 0}

        outcomes = [e.outcome for e in self.buffer]
        wins = sum(1 for o in outcomes if o == 1.0)
        losses = sum(1 for o in outcomes if o == -1.0)
        partial = len(outcomes) - wins - losses

        return {
            "size": len(self.buffer),
            "wins": wins,
            "losses": losses,
            "partial": partial,
            "win_rate": wins / len(outcomes) if outcomes else 0,
            "mean_outcome": sum(outcomes) / len(outcomes) if outcomes else 0,
        }
