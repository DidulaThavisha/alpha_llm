# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AlphaZero-style MCTS + neural network training for competitive programming. A 1.5B LLM (Qwen2.5-Coder) plays "games" against Codeforces problems: each move generates a line of code, the value head predicts win probability, and the model improves through self-play.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Full training loop (self-play → train → evaluate, repeats N times)
python -m scripts.run_full_pipeline --iterations 10
python -m scripts.run_full_pipeline --iterations 10 --mcts-mode deep --mcts-sims 32
python -m scripts.run_full_pipeline --resume latest --hf-repo USER/REPO

# Standalone evaluation
python -m scripts.run_evaluation
python -m scripts.run_evaluation --checkpoint checkpoints/iter_5.pt --use-mcts

# Quick smoke test (loads model, evals 3 problems)
python -m scripts.test_logging
```

On Kaggle/Colab: set `HF_TOKEN` env var, use `--hf-repo` for persistent checkpoints across sessions.

## Architecture

The system has four layers that compose into the AlphaZero loop:

**Model layer** (`model/`): `AlphaCodeModel` wraps Qwen2.5-Coder-1.5B with a bolted-on `ValueHead`. The backbone's lm_head is the policy (next-token logits); the value head is a separate MLP (LayerNorm→Linear→GELU→Linear→Sigmoid) that predicts P(win) from the last token's hidden state. Value head always runs in float32 regardless of backbone dtype to avoid LayerNorm overflow. LoRA (rank=16) is applied to q_proj/v_proj for efficient fine-tuning.

**MCTS layer** (`mcts/`): Line-level search — each "move" generates an entire line of code (up to `\n` or `<eos>`). Two modes dispatched by `MCTSSearch.search()`:
- **Shallow** (1-ply): generate K candidate lines, score each with value head, blend prior×value. Cost: 2K+1 forward passes.
- **Deep** (multi-ply): full select→expand→evaluate→backprop loop with PUCT. Tree grows organically up to `max_search_depth`. Cost: ~N forward passes.

Candidate generation in `generate_candidate_lines()` computes the base KV cache once and samples K lines at different temperatures — no deep KV clone needed.

**Training layer** (`training/`): `SelfPlay` orchestrates games (MCTS generation → code execution → reward). `Trainer` does joint optimization: value loss (MSE with game outcome) + policy loss (KL with MCTS-improved policy). Differential LR: backbone 1e-5, value head 1e-3. Supervised bootstrap seeds ground-truth solutions as "perfect games" (ratio decays 80%→0%). `ReplayBuffer` stores `Experience` dataclasses sampled for training.

**Evaluation layer** (`evaluation/`): `CodeExecutor` runs generated code in subprocess sandbox (10s timeout). Test cases are compressed as base64→zlib→pickle→JSON in the dataset. `compute_reward()` returns +1.0 (all pass), partial credit, or -1.0 (crash).

## Key Design Decisions

- **Line-level, not token-level MCTS**: token-level has 150K branching factor; line-level is tractable
- **`MCTSResult` carries selected_tokens/text**: avoids the old bug of re-expanding root to find the selected child
- **NaN guards everywhere**: value head can return NaN on some devices → `_safe_value()` clamps to 0.5; trainer skips backward pass on NaN loss; gradient NaN check before optimizer step
- **Curriculum**: problems filtered by rating, expanding from easy (800–1200) to hard (800–3500) over iterations
- **HF Hub sync** (`hf_sync.py`): checkpoints + training logs auto-upload; `--resume latest` auto-downloads the most recent checkpoint

## Config

All hyperparameters live in `config.py` as frozen dataclasses (`ModelConfig`, `MCTSConfig`, `TrainingConfig`, `EvalConfig`, `AlphaCodeConfig`). Scripts override via CLI args. The `hf_repo` field in `AlphaCodeConfig` controls persistent storage.

## Conventions

- `logger.py` is a module-level singleton (import as `import logger`, call `logger.console.print(...)`)
- Scripts set `sys.path.insert(0, ...)` and `TOKENIZERS_PARALLELISM=false` at top
- Device auto-detection: MPS → CUDA → CPU
- Checkpoints are `{"iteration", "model_state_dict", "optimizer_state_dict"}` dicts saved via `torch.save`
- The `model.forward()` returns a dict `{"logits", "value", "past_key_values"}`, not separate tensors
