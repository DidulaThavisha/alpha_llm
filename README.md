# AlphaZero for Code Generation

AlphaZero-style MCTS + neural network training applied to competitive programming. Each Codeforces problem is a "game" — the model self-plays, discovers solutions, and improves through the loop.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Full AlphaZero training loop (self-play → train → evaluate)
python -m scripts.run_full_pipeline --iterations 10

# Evaluate current model (greedy, no MCTS)
python -m scripts.run_evaluation

# Evaluate with MCTS search
python -m scripts.run_evaluation --use-mcts --mcts-sims 20

# Resume from checkpoint
python -m scripts.run_full_pipeline --resume checkpoints/iter_5.pt
```

## Architecture

- **Model**: Qwen2.5-Coder-1.5B-Instruct + value head (LoRA fine-tuned)
- **MCTS**: Line-level search (each "move" = one line of code), PUCT selection
- **Training**: Policy loss (KL with MCTS policy) + Value loss (MSE with game outcome)
- **Dataset**: 26 Codeforces problems, rating 800–3400
