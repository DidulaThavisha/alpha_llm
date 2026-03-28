"""Centralized hyperparameters for AlphaZero-style code generation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    precision: str = "float16"
    max_seq_length: int = 2048
    hidden_size: int = 1536  # Qwen2.5-Coder-1.5B hidden dim
    value_head_hidden: int = 512
    value_head_dropout: float = 0.1
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class MCTSConfig:
    num_simulations: int = 8
    c_puct: float = 2.5
    candidate_lines_k: int = 4  # candidates per line decision
    num_beams: int = 4
    top_k_tokens: int = 40  # token-level filtering within beam search
    temperature_early: float = 1.0  # for first N lines
    temperature_late: float = 0.5
    temperature_switch_line: int = 5
    eval_temperature: float = 0.0  # greedy for evaluation
    # Dirichlet noise at root
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    # Progressive widening
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    # Pruning
    value_pruning_threshold: float = 0.01
    # Max generation
    max_lines: int = 50
    max_tokens: int = 1024


@dataclass
class TrainingConfig:
    backbone_lr: float = 1e-5
    value_head_lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 2
    grad_accumulation_steps: int = 4  # effective batch = 8
    epochs_per_iteration: int = 3
    max_iterations: int = 100
    # Replay buffer
    replay_buffer_size: int = 10_000
    # Self-play
    games_per_problem: int = 4
    # Supervised bootstrap (decays over iterations)
    supervised_ratio_start: float = 0.8
    supervised_ratio_end: float = 0.0
    supervised_decay_iterations: int = 30
    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    # Curriculum
    curriculum_stages: List[dict] = field(default_factory=lambda: [
        {"iterations": (0, 5), "max_rating": 1200},
        {"iterations": (5, 15), "max_rating": 2000},
        {"iterations": (15, 100), "max_rating": 3500},
    ])


@dataclass
class EvalConfig:
    code_timeout: int = 10  # seconds
    max_memory_mb: int = 256
    num_eval_games: int = 1  # games per problem during eval (greedy)


@dataclass
class AlphaCodeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Paths
    dataset_path: str = "codeforces_one_per_rating.json"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "auto"  # auto-detect: mps > cuda > cpu


def get_config() -> AlphaCodeConfig:
    return AlphaCodeConfig()
