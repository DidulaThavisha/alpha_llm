"""End-to-end AlphaZero training loop for code generation.

Usage: python -m scripts.run_full_pipeline [--iterations N] [--resume PATH] [--hf-repo REPO]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time
import torch

import logger
from config import get_config, AlphaCodeConfig
from model import AlphaCodeModel
from data.dataset import load_problems, get_problems_by_max_rating
from training.self_play import SelfPlay
from training.trainer import Trainer
from training.replay_buffer import ReplayBuffer


hf_token = os.environ.get("HF_TOKEN")

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    print("Running in Kaggle")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
        except Exception:
            pass


def get_device(config):
    if config.device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return config.device


def get_curriculum_rating(iteration: int, config: AlphaCodeConfig) -> int:
    for stage in config.training.curriculum_stages:
        start, end = stage["iterations"]
        if start <= iteration < end:
            return stage["max_rating"]
    return 3500


def evaluate_model(model, problems, config):
    """Evaluate model on problems using greedy decoding."""
    from mcts.search import MCTSSearch
    from data.prompt_templates import format_prompt, extract_code
    from evaluation.test_case_loader import load_test_cases
    from evaluation.code_executor import CodeExecutor
    from evaluation.reward import compute_reward

    executor = CodeExecutor(timeout=config.eval.code_timeout)
    mcts = MCTSSearch(model, config.mcts)
    results = []

    logger.eval_start(len(problems))

    for problem in problems:
        prompt_text = format_prompt(problem.prompt)
        prompt_ids = model.tokenizer.encode(prompt_text, add_special_tokens=True)
        test_cases = load_test_cases(problem.answer)

        code, _ = mcts.generate_solution(prompt_ids, use_mcts=False)
        code = extract_code(prompt_text + code)
        result = executor.evaluate(code, test_cases)
        reward = compute_reward(result)
        won = reward == 1.0

        logger.eval_problem(problem.title, problem.rating, result.passed, result.total, won, code=code)
        results.append({"rating": problem.rating, "title": problem.title,
                        "passed": result.passed, "total": result.total, "won": won,
                        "code": code})

    wins = sum(1 for r in results if r["won"])
    logger.eval_summary(wins, len(results))
    return results


def setup_hf_sync(config, args):
    """Create HFSync if a repo is configured."""
    repo = getattr(args, "hf_repo", None) or config.hf_repo
    if not repo:
        return None
    from hf_sync import HFSync
    sync = HFSync(repo)
    logger.console.print(f"  HF Hub: [bold]{repo}[/]")
    return sync


def run_pipeline(args):
    config = get_config()
    device = get_device(config)

    # Apply MCTS mode from CLI
    if args.mcts_mode is not None:
        config.mcts.search_mode = args.mcts_mode
    if args.mcts_sims is not None:
        config.mcts.num_simulations = args.mcts_sims
    elif config.mcts.search_mode == "deep" and config.mcts.num_simulations < 16:
        config.mcts.num_simulations = 32  # sensible default for deep

    logger.banner()
    logger.console.print(f"  Device: [bold]{device}[/] · Model: [bold]{config.model.name}[/]")
    logger.console.print(f"  MCTS: [bold]{config.mcts.search_mode}[/] · {config.mcts.num_simulations} sims")

    # HF Hub sync
    hf_sync = setup_hf_sync(config, args)

    logger.console.print("\n[dim]Loading model...[/]")
    model = AlphaCodeModel(config.model)
    model.to_device(device)

    logger.console.print("[dim]Applying LoRA...[/]")
    model.apply_lora()

    replay_buffer = ReplayBuffer(config.training.replay_buffer_size)
    self_play = SelfPlay(model, config)
    trainer = Trainer(model, config)

    # Resume from checkpoint (tries HF Hub if local file missing)
    if args.resume:
        if args.resume == "latest" and hf_sync is not None:
            # Auto-find latest checkpoint from HF
            latest = hf_sync.get_latest_checkpoint()
            if latest:
                ckpt_name = os.path.basename(latest)
                local_path = os.path.join(config.checkpoint_dir, ckpt_name)
                trainer.load_checkpoint(local_path, hf_sync=hf_sync)
            else:
                logger.console.print("[yellow]  No checkpoints found on HF Hub, starting fresh[/]")
        else:
            trainer.load_checkpoint(args.resume, hf_sync=hf_sync)

    all_problems = load_problems(config.dataset_path)
    logger.console.print(f"  Loaded [bold]{len(all_problems)}[/] problems (rating {all_problems[0].rating}–{all_problems[-1].rating})")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    training_log = []
    num_iterations = args.iterations or config.training.max_iterations

    # === MAIN LOOP ===
    for iteration in range(trainer.iteration, num_iterations):
        iter_start = time.time()

        max_rating = get_curriculum_rating(iteration, config)
        problems = get_problems_by_max_rating(all_problems, max_rating)

        logger.iteration_header(iteration + 1, num_iterations, max_rating, len(problems))

        # 1. SUPERVISED BOOTSTRAP
        supervised_ratio = trainer.get_supervised_ratio()
        if supervised_ratio > 0:
            num_supervised = max(1, int(len(problems) * supervised_ratio))
            logger.supervised_info(num_supervised, supervised_ratio)
            for problem in problems[:num_supervised]:
                self_play.supervised_game(problem, replay_buffer)

        # 2. SELF-PLAY
        model.backbone.eval()
        with torch.no_grad():
            self_play_stats = self_play.run_self_play_iteration(
                problems, replay_buffer, config.training.games_per_problem
            )

        # 3. TRAINING
        model.backbone.train()
        epoch_results = trainer.train_iteration(replay_buffer)

        # 4. EVALUATION
        model.backbone.eval()
        with torch.no_grad():
            eval_results = evaluate_model(model, problems, config)

        eval_wins = sum(1 for r in eval_results if r["won"])

        # 5. LOGGING
        iter_time = time.time() - iter_start
        avg_wr = sum(s["win_rate"] for s in self_play_stats) / len(self_play_stats)
        avg_rw = sum(s["mean_reward"] for s in self_play_stats) / len(self_play_stats)

        iter_log = {
            "iteration": iteration + 1,
            "time_seconds": iter_time,
            "curriculum_max_rating": max_rating,
            "num_problems": len(problems),
            "supervised_ratio": supervised_ratio,
            "replay_buffer_size": len(replay_buffer),
            "replay_buffer_stats": replay_buffer.stats,
            "self_play_avg_win_rate": avg_wr,
            "self_play_avg_reward": avg_rw,
            "training_losses": epoch_results,
            "eval_win_rate": eval_wins / len(eval_results),
            "eval_wins": eval_wins,
            "eval_total": len(eval_results),
        }
        training_log.append(iter_log)

        # Save training log locally + HF
        log_path = os.path.join(config.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        if hf_sync is not None:
            hf_sync.upload_log(log_path)

        # Save checkpoint every 5 iterations
        if (iteration + 1) % 5 == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"iter_{iteration+1}.pt")
            trainer.save_checkpoint(ckpt_path, hf_sync=hf_sync)

        logger.iteration_footer(iteration + 1, iter_time, eval_wins, len(eval_results))

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final.pt")
    trainer.save_checkpoint(final_path, hf_sync=hf_sync)

    logger.console.print()
    logger.final_summary(training_log)
    logger.console.print(f"\n[bold green]Training complete![/]")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero-style code generation training")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint, or 'latest' to auto-download from HF Hub")
    parser.add_argument("--mcts-mode", choices=["shallow", "deep"], default=None,
                        help="MCTS search mode: shallow (1-ply, fast) or deep (multi-ply, full MCTS)")
    parser.add_argument("--mcts-sims", type=int, default=None,
                        help="Number of MCTS simulations (default: 8 shallow, 32 deep)")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint sync (e.g. username/alpha-llm)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
