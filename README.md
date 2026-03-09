# Arnold (PyTorch Backend)

This project implements a custom Hebbian "model" architecture and includes:

- `pretrain.py`: pretrains the encoder from a HuggingFace dataset
- `chat_server.py`: FastAPI chat server with SSE token streaming
- `main.py`: lifecycle demo

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pretrain (with training config)

Print default config:

```bash
python pretrain.py --print_default_config
```

Run with defaults:

```bash
python pretrain.py
```

Run with JSON config:

```bash
python pretrain.py --config pretrain_config.json
```

CLI flags override config values when provided.

## Start chat server

```bash
python chat_server.py --storage_dir ./model/pretrained
```

Open `http://localhost:7860`.

## New features (recent additions)

* **Real‑time gradient training** during chat interactions; configure via
  `genome.gradient.lr` or use the pretraining script flags `--gradient_lr`,
  `--gradient_steps`, `--gradient_batch_size`, and `--freeze_encoder`.
* **Evaluation harness** (`evaluate.py`) that computes perplexity and logs to
  TensorBoard.  Useful for measuring progress and regression testing.
* **Hyperparameter sweep** example (`run_experiment.py`) using Optuna.
* **Improved checkpoint metadata** now includes the RNG seed for reproducibility.
* **Unit tests** available under `tests/`; run them with `pytest`.
* **Architecture docs** in `docs/architecture.md` with a Mermaid diagram.

## Running the test suite

After installing development dependencies (`pip install -r requirements.txt`)
execute:

```bash
pytest tests/
```
