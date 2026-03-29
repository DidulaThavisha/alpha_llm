"""Evaluate the model on all problems using greedy decoding (no MCTS).

Usage: python -m scripts.run_evaluation [--checkpoint PATH] [--hf-repo REPO]
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

        logger.eval_problem(problem.title, problem.rating, result.passed, result.total, won, code=code)
        results.append({"rating": problem.rating, "title": problem.title,
                        "passed": result.passed, "total": result.total,
                        "reward": reward, "won": won, "code": code})

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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint, or 'latest' to auto-download from HF Hub")
    parser.add_argument("--use-mcts", action="store_true")
    parser.add_argument("--mcts-sims", type=int, default=10)
    parser.add_argument("--mcts-mode", choices=["shallow", "deep"], default="shallow",
                        help="MCTS search mode: shallow (1-ply) or deep (multi-ply)")
    parser.add_argument("--max-rating", type=int, default=3500)
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint download")
    args = parser.parse_args()

    config = get_config()
    config.mcts.search_mode = args.mcts_mode
    device = get_device(config)

    # HF Hub sync
    hf_sync = None
    repo = args.hf_repo or config.hf_repo
    if repo:
        from hf_sync import HFSync
        hf_sync = HFSync(repo)

    logger.banner()
    logger.console.print(f"  Device: [bold]{device}[/]")

    logger.console.print("[dim]Loading model...[/]")
    model = AlphaCodeModel(config.model)
    model.to_device(device)
    model.backbone.eval()

    if args.checkpoint:
        if args.checkpoint == "latest" and hf_sync is not None:
            latest = hf_sync.get_latest_checkpoint()
            if latest:
                ckpt_name = os.path.basename(latest)
                local_path = os.path.join(config.checkpoint_dir, ckpt_name)
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                hf_sync.download_checkpoint(ckpt_name, local_path)
                checkpoint = torch.load(local_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.console.print(f"  [dim]Loaded checkpoint: {ckpt_name} from HF Hub[/]")
            else:
                logger.console.print("[yellow]  No checkpoints found on HF Hub[/]")
        else:
            ckpt_path = args.checkpoint
            # Try HF download if local file doesn't exist
            if not os.path.exists(ckpt_path) and hf_sync is not None:
                name = os.path.basename(ckpt_path)
                os.makedirs(os.path.dirname(ckpt_path) or config.checkpoint_dir, exist_ok=True)
                hf_sync.download_checkpoint(name, ckpt_path)

            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.console.print(f"  [dim]Loaded checkpoint: {ckpt_path}[/]")

    problems = load_problems(config.dataset_path)
    problems = [p for p in problems if p.rating <= args.max_rating]

    mode = "MCTS" if args.use_mcts else "greedy"
    logger.console.print(f"  {len(problems)} problems · mode: [bold]{mode}[/]")

    with torch.no_grad():
        evaluate_model(model, problems, config, args.use_mcts, args.mcts_sims)


if __name__ == "__main__":
    main()
