"""Rich logging for the AlphaZero training pipeline."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text
from rich import box
from typing import List, Dict, Any

console = Console()


def banner():
    console.print(Panel.fit(
        "[bold cyan]AlphaZero for Code Generation[/]\n"
        "[dim]MCTS + Neural Network · Self-Play · Competitive Programming[/]",
        border_style="cyan",
    ))


def iteration_header(iteration: int, total: int, max_rating: int, num_problems: int):
    console.rule(f"[bold yellow]Iteration {iteration}/{total}[/]")
    console.print(f"  Curriculum: [green]{num_problems}[/] problems · max rating [green]{max_rating}[/]")


def self_play_start(num_problems: int, games_per: int):
    console.print(f"\n[bold blue]▶ Self-Play[/]  {num_problems} problems × {games_per} games")


def self_play_problem(title: str, rating: int, win_rate: float, mean_reward: float, unique: int):
    color = "green" if win_rate > 0.5 else "yellow" if win_rate > 0 else "red"
    console.print(
        f"  [{color}]{'●' if win_rate > 0 else '○'}[/] "
        f"[bold]{title}[/] [dim](rating {rating})[/]  "
        f"win={win_rate:.0%}  reward={mean_reward:+.2f}  "
        f"unique={unique}"
    )


def self_play_summary(stats: List[Dict[str, Any]], buffer_size: int):
    avg_wr = sum(s["win_rate"] for s in stats) / len(stats) if stats else 0
    avg_rw = sum(s["mean_reward"] for s in stats) / len(stats) if stats else 0
    total_unique = sum(s["unique_solutions"] for s in stats)
    console.print(
        f"  [dim]───[/] avg win rate [bold]{avg_wr:.0%}[/] · "
        f"avg reward [bold]{avg_rw:+.2f}[/] · "
        f"unique solutions [bold]{total_unique}[/] · "
        f"buffer [bold]{buffer_size}[/]"
    )


def training_start(epochs: int, buffer_size: int):
    console.print(f"\n[bold magenta]▶ Training[/]  {epochs} epochs · {buffer_size} experiences")


def training_epoch(epoch: int, total: int, value_loss: float, policy_loss: float):
    bar = "█" * int(epoch / total * 20)
    empty = "░" * (20 - len(bar))
    console.print(
        f"  [{bar}{empty}] epoch {epoch}/{total}  "
        f"value_loss=[cyan]{value_loss:.4f}[/]  policy_loss=[cyan]{policy_loss:.4f}[/]"
    )


def eval_start(num_problems: int):
    console.print(f"\n[bold green]▶ Evaluation[/]  {num_problems} problems (greedy)")


def eval_problem(title: str, rating: int, passed: int, total: int, won: bool):
    icon = "[green]✓[/]" if won else "[red]✗[/]"
    console.print(f"  {icon} [bold]{title}[/] [dim](rating {rating})[/]  {passed}/{total} tests")


def eval_summary(wins: int, total: int):
    rate = wins / total if total > 0 else 0
    color = "green" if rate > 0.5 else "yellow" if rate > 0 else "red"
    console.print(Panel(
        f"[bold {color}]{wins}/{total} solved ({rate:.0%})[/]",
        title="Evaluation Result",
        border_style=color,
        expand=False,
    ))


def iteration_footer(iteration: int, time_seconds: float, eval_wins: int, eval_total: int):
    console.print(
        f"\n  [dim]Iteration {iteration} completed in {time_seconds:.0f}s · "
        f"solved {eval_wins}/{eval_total}[/]\n"
    )


def checkpoint_saved(path: str):
    console.print(f"  [dim]💾 Checkpoint saved: {path}[/]")


def supervised_info(count: int, ratio: float):
    console.print(f"\n[bold]▶ Supervised Bootstrap[/]  {count} ground-truth games (ratio {ratio:.0%})")


def game_result(game_num: int, reward: float, code_preview: str):
    color = "green" if reward == 1.0 else "yellow" if reward > -0.5 else "red"
    console.print(f"    game {game_num}: [{color}]reward={reward:+.2f}[/]  [dim]{code_preview[:60]}[/]")


def final_summary(log: List[Dict[str, Any]]):
    if not log:
        return

    table = Table(title="Training Summary", box=box.ROUNDED)
    table.add_column("Iter", style="cyan", justify="right")
    table.add_column("Problems", justify="right")
    table.add_column("Self-Play WR", justify="right")
    table.add_column("Eval Solved", justify="right")
    table.add_column("Value Loss", justify="right")
    table.add_column("Time", justify="right")

    for entry in log:
        losses = entry.get("training_losses", [{}])
        vloss = losses[-1].get("value_loss", 0) if losses else 0
        table.add_row(
            str(entry["iteration"]),
            str(entry["num_problems"]),
            f"{entry['self_play_avg_win_rate']:.0%}",
            f"{entry['eval_wins']}/{entry['eval_total']}",
            f"{vloss:.4f}",
            f"{entry['time_seconds']:.0f}s",
        )

    console.print(table)
