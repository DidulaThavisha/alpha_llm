"""Rich logging for the AlphaZero training pipeline."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text
from rich import box
from typing import List, Dict, Any, Optional

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


# ── Self-Play ──────────────────────────────────────────────────────

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


# ── Game-level logging (generated code + test results) ─────────────

def game_result(game_num: int, reward: float, code_preview: str):
    color = "green" if reward == 1.0 else "yellow" if reward > -0.5 else "red"
    console.print(f"    game {game_num}: [{color}]reward={reward:+.2f}[/]  [dim]{code_preview[:60]}[/]")


def game_generation(game_num: int, problem_title: str, code: str, reward: float,
                    passed: int, total: int, errors: Optional[List[str]] = None,
                    trajectory_len: int = 0, time_seconds: float = 0.0):
    """Log the full details of a single self-play game."""
    won = reward == 1.0
    status_color = "green" if won else "yellow" if reward > -0.5 else "red"
    status_text = "WIN" if won else f"FAIL ({passed}/{total})"

    header = (
        f"[{status_color} bold]Game {game_num}[/] · "
        f"[bold]{problem_title}[/] · "
        f"[{status_color}]{status_text}[/] · "
        f"reward={reward:+.2f} · "
        f"lines={trajectory_len} · "
        f"{time_seconds:.1f}s"
    )
    console.print(f"    {header}")

    # Show generated code
    if code.strip():
        trimmed = code if len(code) < 1500 else code[:1500] + "\n# ... (truncated)"
        syntax = Syntax(trimmed, "python", theme="monokai", line_numbers=True,
                        word_wrap=True, padding=(0, 1))
        console.print(Panel(syntax, title=f"[dim]Generated Code[/]",
                            border_style="dim", expand=False, width=min(100, console.width - 4)))

    # Show test results
    if total > 0:
        bar_len = 30
        filled = int(passed / total * bar_len)
        bar = f"[green]{'█' * filled}[/][red]{'░' * (bar_len - filled)}[/]"
        console.print(f"      Tests: {bar} {passed}/{total}")

    # Show first few errors if any
    if errors:
        for i, err in enumerate(errors[:3]):
            err_short = err[:120].replace("\n", " ")
            console.print(f"      [red dim]error {i+1}:[/] [dim]{err_short}[/]")
        if len(errors) > 3:
            console.print(f"      [dim]... and {len(errors)-3} more errors[/]")


def mcts_search_step(line_num: int, num_children: int, best_value: float,
                     simulations: int, selected_line: str):
    """Log a single MCTS decision point."""
    console.print(
        f"        [dim]line {line_num}:[/] "
        f"{num_children} candidates · "
        f"value={best_value:.3f} · "
        f"{simulations} sims · "
        f"[cyan]{selected_line.rstrip()[:70]}[/]"
    )


# ── Training ───────────────────────────────────────────────────────

def training_start(epochs: int, buffer_size: int):
    console.print(f"\n[bold magenta]▶ Training[/]  {epochs} epochs · {buffer_size} experiences")


def training_epoch(epoch: int, total: int, value_loss: float, policy_loss: float):
    bar = "█" * int(epoch / total * 20)
    empty = "░" * (20 - len(bar))
    console.print(
        f"  [{bar}{empty}] epoch {epoch}/{total}  "
        f"value_loss=[cyan]{value_loss:.4f}[/]  policy_loss=[cyan]{policy_loss:.4f}[/]"
    )


# ── Evaluation ─────────────────────────────────────────────────────

def eval_start(num_problems: int):
    console.print(f"\n[bold green]▶ Evaluation[/]  {num_problems} problems (greedy)")


def eval_problem(title: str, rating: int, passed: int, total: int, won: bool,
                 code: Optional[str] = None):
    icon = "[green]✓[/]" if won else "[red]✗[/]"
    console.print(f"  {icon} [bold]{title}[/] [dim](rating {rating})[/]  {passed}/{total} tests")

    # Show code for failed evaluations so we can see what went wrong
    if not won and code and code.strip():
        trimmed = code if len(code) < 800 else code[:800] + "\n# ... (truncated)"
        syntax = Syntax(trimmed, "python", theme="monokai", line_numbers=True,
                        word_wrap=True, padding=(0, 1))
        console.print(Panel(syntax, title=f"[dim]Generated ({passed}/{total} pass)[/]",
                            border_style="red dim", expand=False, width=min(90, console.width - 6)))


def eval_summary(wins: int, total: int):
    rate = wins / total if total > 0 else 0
    color = "green" if rate > 0.5 else "yellow" if rate > 0 else "red"
    console.print(Panel(
        f"[bold {color}]{wins}/{total} solved ({rate:.0%})[/]",
        title="Evaluation Result",
        border_style=color,
        expand=False,
    ))


# ── Misc ───────────────────────────────────────────────────────────

def iteration_footer(iteration: int, time_seconds: float, eval_wins: int, eval_total: int):
    console.print(
        f"\n  [dim]Iteration {iteration} completed in {time_seconds:.0f}s · "
        f"solved {eval_wins}/{eval_total}[/]\n"
    )


def checkpoint_saved(path: str):
    console.print(f"  [dim]💾 Checkpoint saved: {path}[/]")


def supervised_info(count: int, ratio: float):
    console.print(f"\n[bold]▶ Supervised Bootstrap[/]  {count} ground-truth games (ratio {ratio:.0%})")


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
