# Arnold (PyTorch Backend)

This project implements a custom Hebbian "brain" architecture and includes:

- `pretrain.py`: pretrains the brainstem from a HuggingFace dataset
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
