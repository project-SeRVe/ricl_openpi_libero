# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RICL (Re-training for In-Context Learning) version of the openpi repository, focused on RICL-Pi0-FAST-DROID — a vision-language-action (VLA) model for robotic control. Built on JAX/Flax with LeRobot datasets and Hugging Face Transformers. Targets Franka DROID robots, with additional support for ALOHA and LIBERO.

## Build & Development Commands

```bash
# Install dependencies (uv is the package manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss google-genai openai

# Run all CI-friendly tests
uv run pytest -m "not manual"

# Run a single test file
uv run pytest src/openpi/models/model_test.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Pre-commit hooks (uv-lock + ruff)
pre-commit run --all-files
```

## Training & Serving

```bash
# Train RICL model
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl --exp-name=EXPERIMENT --overwrite

# Fine-tune on new task
python scripts/train_pi0_fast_ricl.py pi0_fast_droid_ricl___finetune_on_new_task --exp-name=EXPERIMENT --overwrite

# Serve a checkpoint
uv run scripts/serve_policy_ricl.py policy:checkpoint \
  --policy.config=pi0_fast_droid_ricl \
  --policy.dir=checkpoints/pi0_fast_droid_ricl/EXPERIMENT/5400 \
  --policy.demos_dir=preprocessing/collected_demos/{YYYY-MM-DD}_{task_prompt}
```

## Data Preprocessing

```bash
cd preprocessing
python process_collected_demos.py --dir_of_dirs=collected_demos_training
python retrieve_within_collected_demo_groups.py
python process_collected_demos.py --dir_of_dirs=collected_demos
```

Demo folders must follow the naming convention `{YYYY-MM-DD}_{task_prompt}` with underscores separating words in the prompt. Each demo folder must contain `traj.h5` and `recordings/frames/` with camera subdirectories (`hand_camera/`, `varied_camera_1/`, `varied_camera_2/`).

## Architecture

### Core source (`src/openpi/`)
- **models/**: ML model implementations. `pi0_fast_ricl.py` is the primary RICL model with attention mask generation for in-context learning and retrieval-augmented prompting. Built on `gemma.py`/`gemma_fast.py` (language) and `siglip.py`/`vit.py` (vision). LoRA fine-tuning in `lora.py`.
- **policies/**: Robot-specific policy wrappers (`droid_policy.py`, `aloha_policy.py`, `libero_policy.py`) over the base `policy.py` interface.
- **training/**: Training infrastructure — `config.py` defines all model/data configurations (look here for `pi0_fast_droid_ricl` and `pi0_fast_droid_ricl___finetune_on_new_task`), `data_loader.py` handles LeRobot datasets, `sharding.py` manages JAX distributed training.
- **transforms.py**: Data transformation pipeline (normalization, augmentation, tokenization).
- **serving/**: `websocket_policy_server.py` for remote inference over WebSocket.

### Client library (`packages/openpi-client/`)
Lightweight client with minimal dependencies. `WebsocketClientPolicy` connects to the serving endpoint. Includes a runtime/agent system for deployment.

### Entry points (`scripts/`)
- `train_pi0_fast_ricl.py`: RICL training with in-context demo handling
- `serve_policy_ricl.py`: RICL policy server loading checkpoint + retrieval demos
- `setup_norm_states_for_ricl.py`: Compute normalization stats for RICL training data

### Robot integration (`examples/`)
- `examples/droid/main_ricl.py`: Client-side DROID control with RICL
- `examples/droid/main_ricl_finetune.py`: Fine-tuned policy control

## Code Style

- Python 3.11+, 4-space indentation, 120 char line length
- Ruff for linting and formatting (configured in `pyproject.toml`)
- `snake_case` for files/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Tests as `*_test.py` beside the code they validate; mark long-running tests with `@pytest.mark.manual`

## Important Paths (gitignored)

`checkpoints/`, `wandb/`, `preprocessing/collected_demos_training/`, `preprocessing/collected_demos/` — never commit these.
