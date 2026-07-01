from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import crypten

from crypten_rank_bench import crypten_security_profile, load_bert_model_and_tokenizer, run_bert_full_once


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def request_samples(req: dict) -> list[dict]:
    samples = req.get("samples")
    if samples is None:
        text = str(req.get("text", "This movie is wonderful and heartwarming."))
        samples = [{"text": text, "label": req.get("label")}]
    out = []
    for item in samples:
        if isinstance(item, str):
            out.append({"text": item, "label": None})
        else:
            out.append({"text": str(item["text"]), "label": item.get("label")})
    return out


def main() -> int:
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 2)
    service_dir = Path(os.environ.get("CRYPTEN_SERVICE_DIR", "/workspace/out/crypten_service"))
    req_dir = service_dir / "requests"
    resp_dir = service_dir / "responses"
    stop_path = service_dir / "stop"
    ready_path = service_dir / f"rank{rank}.ready"
    heartbeat_path = service_dir / f"rank{rank}.heartbeat"
    req_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)

    max_seq_len = env_int("CRYPTEN_MAX_SEQ_LEN", 16)
    max_layers = env_int("CRYPTEN_MAX_LAYERS", 12)
    poll_s = float(os.environ.get("CRYPTEN_SERVICE_POLL_S", "0.05"))

    print(f"[crypten-service-rank{rank}] initializing world_size={world_size}", flush=True)
    crypten.init()
    model, tokenizer = load_bert_model_and_tokenizer()
    ready = {
        "rank": rank,
        "world_size": world_size,
        "state": "ready",
        "max_seq_len": max_seq_len,
        "max_layers": max_layers,
        "rendezvous": os.environ.get("RENDEZVOUS", ""),
        "backend": os.environ.get("DISTRIBUTED_BACKEND", ""),
        "ready_at": time.time(),
    }
    write_json_atomic(ready_path, ready)
    print(f"[crypten-service-rank{rank}] ready", flush=True)

    seen: set[str] = set()
    request_count = 0
    try:
        while not stop_path.exists():
            heartbeat_path.write_text(str(time.time()), encoding="utf-8")
            requests = sorted(req_dir.glob("*.json"), key=lambda p: p.name)
            next_path = None
            for path in requests:
                if path.stem not in seen:
                    next_path = path
                    break
            if next_path is None:
                time.sleep(poll_s)
                continue

            req_id = next_path.stem
            req = read_json(next_path)
            seen.add(req_id)
            if req.get("command") == "shutdown":
                break

            samples = request_samples(req)
            t0 = time.perf_counter()
            result = run_bert_full_once(model, tokenizer, samples)
            elapsed = time.perf_counter() - t0
            request_count += 1
            payload = {
                "rank": rank,
                "world_size": world_size,
                "request_id": req_id,
                "state": "complete",
                "elapsed_s": elapsed,
                "max_seq_len": max_seq_len,
                "max_layers": max_layers,
                "n_samples": len(result.get("predictions", [])),
                "predictions": result.get("predictions", []),
                "sample_times_s": result.get("sample_times_s", []),
                "plain_sample_times_s": result.get("plain_sample_times_s", []),
                "plain_median_s": statistics.median(result.get("plain_sample_times_s", []))
                if result.get("plain_sample_times_s")
                else None,
                "sum_last": result.get("sum_last"),
                "security_profile": crypten_security_profile("bert_full"),
                "request_count": request_count,
            }
            write_json_atomic(resp_dir / f"{req_id}.rank{rank}.json", payload)
            if rank == 0:
                write_json_atomic(resp_dir / f"{req_id}.json", payload)
            print(f"[crypten-service-rank{rank}] request {req_id} done elapsed_s={elapsed:.6f}", flush=True)
    finally:
        crypten.uninit()
        ready_path.unlink(missing_ok=True)
        heartbeat_path.unlink(missing_ok=True)
        print(f"[crypten-service-rank{rank}] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
