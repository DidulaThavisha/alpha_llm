"""Self-play: generate code solutions using MCTS and collect trajectories.

Each "game" is one attempt to solve a problem:
1. Format the prompt
2. Use MCTS to generate code line-by-line
3. Execute the code against test cases
4. Assign reward
5. Store trajectory in replay buffer
"""

import torch
from typing import List, Dict, Any, Tuple, Optional

import logger
from data.dataset import CodeProblem
from data.prompt_templates import format_prompt, extract_code
from evaluation.test_case_loader import load_test_cases
from evaluation.code_executor import CodeExecutor
from evaluation.reward import compute_reward
from mcts.search import MCTSSearch
from training.replay_buffer import ReplayBuffer, Experience
from config import AlphaCodeConfig


class SelfPlay:
    """Orchestrates MCTS-guided self-play for code generation."""

    def __init__(self, model, config: AlphaCodeConfig):
        self.model = model
        self.config = config
        self.mcts = MCTSSearch(model, config.mcts)
        self.executor = CodeExecutor(
            timeout=config.eval.code_timeout,
            max_memory_mb=config.eval.max_memory_mb,
        )

    def play_one_game(
        self,
        problem: CodeProblem,
        use_mcts: bool = True,
    ) -> Tuple[float, str, List[Dict[str, Any]]]:
        """Play one self-play game on a problem."""
        prompt_text = format_prompt(problem.prompt)
        tokenizer = self.model.tokenizer
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        generated_code, trajectory = self.mcts.generate_solution(
            prompt_ids=prompt_ids,
            use_mcts=use_mcts,
        )

        code = extract_code(prompt_text + generated_code)
        if not code:
            code = generated_code.strip()

        test_cases = load_test_cases(problem.answer)
        result = self.executor.evaluate(code, test_cases)
        reward = compute_reward(result)

        return reward, code, trajectory

    def play_games(
        self,
        problem: CodeProblem,
        replay_buffer: ReplayBuffer,
        num_games: int,
    ) -> Dict[str, Any]:
        """Play multiple self-play games on one problem."""
        rewards = []
        codes = []
        unique_solutions = set()

        tokenizer = self.model.tokenizer
        prompt_text = format_prompt(problem.prompt)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        for game_idx in range(num_games):
            reward, code, trajectory = self.play_one_game(problem, use_mcts=True)
            rewards.append(reward)
            codes.append(code)

            if reward == 1.0:
                unique_solutions.add(code.strip())

            logger.game_result(game_idx + 1, reward, code.split("\n")[0] if code else "")

            replay_buffer.add_trajectory(
                trajectory=trajectory,
                prompt_ids=prompt_ids,
                outcome=reward,
                problem_rating=problem.rating,
            )

        wins = sum(1 for r in rewards if r == 1.0)
        return {
            "problem": problem.title,
            "rating": problem.rating,
            "games": num_games,
            "wins": wins,
            "win_rate": wins / num_games,
            "mean_reward": sum(rewards) / len(rewards),
            "unique_solutions": len(unique_solutions),
            "rewards": rewards,
        }

    def run_self_play_iteration(
        self,
        problems: List[CodeProblem],
        replay_buffer: ReplayBuffer,
        games_per_problem: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run self-play on all problems in the current curriculum."""
        gpg = games_per_problem or self.config.training.games_per_problem
        all_stats = []

        logger.self_play_start(len(problems), gpg)

        for problem in problems:
            stats = self.play_games(problem, replay_buffer, gpg)
            all_stats.append(stats)
            logger.self_play_problem(
                stats["problem"], stats["rating"],
                stats["win_rate"], stats["mean_reward"], stats["unique_solutions"],
            )

        logger.self_play_summary(all_stats, len(replay_buffer))
        return all_stats

    def supervised_game(
        self,
        problem: CodeProblem,
        replay_buffer: ReplayBuffer,
    ):
        """Create a supervised training example from the ground-truth solution."""
        tokenizer = self.model.tokenizer
        prompt_text = format_prompt(problem.prompt)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        lines = problem.python_solution.replace("\r\n", "\n").split("\n")
        code_tokens_so_far = []

        trajectory = []
        for line in lines:
            line_with_newline = line + "\n"
            line_tokens = tokenizer.encode(line_with_newline, add_special_tokens=False)

            trajectory.append({
                "state_tokens": code_tokens_so_far.copy(),
                "mcts_policy": {0: 1.0},
                "selected_line": line_with_newline,
                "value_estimate": 1.0,
            })

            code_tokens_so_far.extend(line_tokens)

        replay_buffer.add_trajectory(
            trajectory=trajectory,
            prompt_ids=prompt_ids,
            outcome=1.0,
            problem_rating=problem.rating,
        )
