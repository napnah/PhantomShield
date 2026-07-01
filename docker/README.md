# Docker Runners

This directory contains the container definitions used by the current AegisX experiments and dashboard.

## Roles

- MCU: `mcu-hp`, `mcu-p0`, `mcu-p1`, and `mcu-verify` on a Docker bridge network.
- CrypTen: `crypten-r0` and `crypten-r1` with Gloo/TCP rendezvous.
- Dashboard backend: started on the host, then calls these Docker runners through Python orchestration scripts.

## Build

Run from the repository root:

```bash
docker compose -f docker/docker-compose.mpc.yml build crypten-r0 mcu-hp
docker build -f docker/Dockerfile.mcu.bert_session -t phantomshield-mcu:bert-session-fixed .
```

The compose build creates `phantomshield-crypten:local` and the default MCU image. The full BERT MCU session uses the `phantomshield-mcu:bert-session-fixed` tag.

## Portable model mounts

The BERT runners need:

```text
bert-base-uncased/
checkpoints/bert-sst2/
```

Place them at the repository root or point the runners to external locations:

```bash
export PHANTOM_BERT_BASE_HOST=/abs/path/bert-base-uncased
export PHANTOM_CHECKPOINTS_HOST=/abs/path/checkpoints
```

PowerShell:

```powershell
$env:PHANTOM_BERT_BASE_HOST = "D:\models\bert-base-uncased"
$env:PHANTOM_CHECKPOINTS_HOST = "D:\models\checkpoints"
```

## Main commands

Operator-level Docker comparison:

```bash
python experiments/docker_real_comm/run_docker_comparison.py --preset bert --repeat 3 --batches 1,2,4
```

CrypTen full BERT Docker run:

```bash
python experiments/docker_bert_full/run_docker_bert_comparison.py \
  --samples 1 \
  --repeat 1 \
  --max-seq-len 16 \
  --max-layers 12 \
  --crypten-nonlinear native \
  --skip-build
```

MCU full BERT Docker run:

```bash
python experiments/docker_bert_full/run_mcu_bert_session_docker.py \
  --image phantomshield-mcu:bert-session-fixed \
  --batch 1 \
  --seq 16 \
  --hidden 768 \
  --heads 12 \
  --ffn 3072 \
  --layers 12 \
  --state-mode chained \
  --input-mode real_io \
  --text "This movie is wonderful and heartwarming." \
  --scale-bits 16 \
  --rescale-mode hp_clear
```

Dashboard three-path Docker comparison:

```bash
python - <<'PY'
import sys
sys.path[:0] = ["dashboard/backend", "."]
from bert_orchestrator import compare
out = compare("This movie is wonderful and heartwarming.", launch="docker", max_seq_len=16)
print(out["success"], out["success_count"], out["total_count"])
for item in out["results"]:
    print(item["mode"], item["label"], item["elapsed_seconds"], item.get("artifact_dir"))
PY
```

## Warm service mode

Warm service mode keeps Docker containers alive. Two levels are currently available:

- `docker_warm`: containers stay up; each request still starts a CrypTen rank process or MCU `bert_session` process inside the container.
- `docker_service`: CrypTen v2 keeps rank processes and the BERT model/tokenizer alive across requests. MCU v2.1 keeps `hp/p0/p1` role wrapper processes and a host input-share exporter alive, caches static model shares, and accepts repeated request files; the inner protocol connection and Rust-side tensor loading still happen per request.

```bash
python experiments/docker_bert_full/warm_docker_service.py start
python experiments/docker_bert_full/warm_docker_service.py start-crypten-service
python experiments/docker_bert_full/warm_docker_service.py start-mcu-service
python experiments/docker_bert_full/warm_docker_service.py status
python experiments/docker_bert_full/warm_docker_service.py crypten-service-status
python experiments/docker_bert_full/warm_docker_service.py mcu-service-status
python experiments/docker_bert_full/warm_docker_service.py infer-crypten --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-crypten-service --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-mcu --text "This movie is wonderful and heartwarming."
python experiments/docker_bert_full/warm_docker_service.py infer-mcu-service --text "This movie is wonderful and heartwarming."
```

Backend API:

```bash
curl -X POST http://127.0.0.1:8000/api/bert/docker-service \
  -H "Content-Type: application/json" \
  -d "{\"action\":\"start\"}"
```

## Cleanup

```bash
docker compose -f docker/docker-compose.mpc.yml down --remove-orphans
docker compose -p aegisxwarm -f docker/docker-compose.mpc.yml -f docker/docker-compose.warm.yml down --remove-orphans
```
