# Environment Notes

The project should be deployable without relying on one developer's local machine layout. Prefer Docker for CrypTen and MCU execution, and keep the host Python environment lightweight for orchestration and the dashboard backend.

## Host Python

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Models

Download into the repository root:

```bash
python scripts/download_models.py
```

Or use external model paths for Docker runners:

```bash
export PHANTOM_BERT_BASE_HOST=/abs/path/bert-base-uncased
export PHANTOM_CHECKPOINTS_HOST=/abs/path/checkpoints
```

PowerShell:

```powershell
$env:PHANTOM_BERT_BASE_HOST = "D:\models\bert-base-uncased"
$env:PHANTOM_CHECKPOINTS_HOST = "D:\models\checkpoints"
```

## Dashboard

```bash
python dashboard/backend/main.py
```

Open `dashboard/frontend/index.html`. The BERT sentiment scenario supports process launch and Docker launch, plus a three-path comparison for plaintext, CrypTen, and MCU.

## Docker

```bash
docker compose -f docker/docker-compose.mpc.yml build crypten-r0 mcu-hp
docker build -f docker/Dockerfile.mcu.bert_session -t phantomshield-mcu:bert-session-fixed .
```

See `docker/README.md` and top-level `README.md` for the current full command list.
