"""Evaluate the model on all problems using greedy decoding (no MCTS).

Usage: python -m scripts.run_evaluation [--checkpoint PATH]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch

import logger
from config import get_config
from model import AlphaCodeModel
from data.dataset import load_problems
from data.prompt_templates import format_prompt, extract_code
from evaluation.test_case_loader import load_test_cases
from evaluation.code_executor import CodeExecutor
from evaluation.reward import compute_reward
from mcts.search import MCTSSearch


def evaluate_model(model, problems, config, use_mcts=False, mcts_sims=10):
    executor = CodeExecutor(timeout=config.eval.code_timeout)
    mcts_config = config.mcts
    if use_mcts:
        mcts_config.num_simulations = mcts_sims
    mcts = MCTSSearch(model, mcts_config)

    results = []
    logger.eval_start(len(problems))

    for problem in problems:
        prompt_text = format_prompt(problem.prompt)
        prompt_ids = model.tokenizer.encode(prompt_text, add_special_tokens=True)
        test_cases = load_test_cases(problem.answer)

        code, _ = mcts.generate_solution(prompt_ids, use_mcts=use_mcts)
        code = extract_code(prompt_text + code)
        result = executor.evaluate(code, test_cases)
        reward = compute_reward(result)
        won = reward == 1.0

        logger.eval_problem(problem.title, problem.rating, result.passed, result.total, won)
        results.append({"rating": problem.rating, "title": problem.title,
                        "passed": result.passed, "total": result.total,
                        "reward": reward, "won": won})

    wins = sum(1 for r in results if r["won"])
    logger.eval_summary(wins, len(results))
    return results


def get_device(config):
    if config.device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return config.device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use-mcts", action="store_true")
    parser.add_argument("--mcts-sims", type=int, default=10)
    parser.add_argument("--max-rating", type=int, default=3500)
    args = parser.parse_args()

    config = get_config()
    device = get_device(config)

    logger.banner()
    logger.console.print(f"  Device: [bold]{device}[/]")

    logger.console.print("[dim]Loading model...[/]")
    model = AlphaCodeModel(config.model)
    model.to_device(device)
    model.backbone.eval()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.console.print(f"  [dim]Loaded checkpoint: {args.checkpoint}[/]")

    problems = load_problems(config.dataset_path)
    problems = [p for p in problems if p.rating <= args.max_rating]

    mode = "MCTS" if args.use_mcts else "greedy"
    logger.console.print(f"  {len(problems)} problems · mode: [bold]{mode}[/]")

    with torch.no_grad():
        evaluate_model(model, problems, config, args.use_mcts, args.mcts_sims)


if __name__ == "__main__":
    main()
