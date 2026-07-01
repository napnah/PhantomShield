from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import suppress
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.docker_bert_full.export_bert_session_shares import (  # noqa: E402
    write_csv,
    write_public_u64_tensor,
    write_shared_tensor,
)
from transformer.bert_weight_loader import load_classification_model, load_tokenizer  # noqa: E402
from transformer.plaintext_bert import embed_inputs  # noqa: E402


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_error: OSError | None = None
    for _ in range(5):
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.02)
    raise last_error or OSError(f"failed to write {path}")


def write_heartbeat(path: Path) -> None:
    try:
        path.write_text(f"{time.time():.6f}\n", encoding="utf-8")
    except OSError:
        # Heartbeats are advisory. A transient Windows file lock should not kill
        # the long-lived exporter and leave a stale ready marker behind.
        pass


def export_input(model, tokenizer, text: str, out: Path, max_seq_len: int, scale_bits: int) -> None:
    p0_dir = out / "p0"
    p1_dir = out / "p1"
    p0_dir.mkdir(parents=True, exist_ok=True)
    p1_dir.mkdir(parents=True, exist_ok=True)
    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        hidden = embed_inputs(model, inputs)
    rows: list[dict] = []
    write_shared_tensor("hidden", hidden.reshape(-1, hidden.shape[-1]), p0_dir, p1_dir, rows, 0xBEE70001, scale_bits)
    write_public_u64_tensor("attention_mask", inputs["attention_mask"], p0_dir, p1_dir, rows)
    write_csv(out / "tensor_manifest.csv", rows)
    manifest = {
        "purpose": "Persistent MCU input-share export for bert_session real_io mode.",
        "export_mode": "input_only_service",
        "text": text,
        "max_seq_len": max_seq_len,
        "scale_bits": scale_bits,
        "shape": {
            "batch": int(hidden.shape[0]),
            "seq": int(hidden.shape[1]),
            "hidden": int(hidden.shape[2]),
        },
        "implemented_tensors": [row["name"] for row in rows],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    service_dir = Path(args.service_dir)
    request_dir = service_dir / "input_requests"
    response_dir = service_dir / "input_responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    stop_path = service_dir / "input_exporter.stop"
    ready_path = service_dir / "input_exporter.ready"
    heartbeat_path = service_dir / "input_exporter.heartbeat"

    model = load_classification_model(args.device)
    tokenizer = load_tokenizer()
    write_text_atomic(ready_path, "ready\n")
    seen: set[str] = set()
    try:
        while not stop_path.exists():
            write_heartbeat(heartbeat_path)
            for req_path in sorted(request_dir.glob("*.json")):
                request_id = req_path.stem
                if request_id in seen:
                    continue
                seen.add(request_id)
                done_path = response_dir / f"{request_id}.done"
                error_path = response_dir / f"{request_id}.error"
                t0 = time.perf_counter()
                try:
                    payload = json.loads(req_path.read_text(encoding="utf-8"))
                    export_input(
                        model,
                        tokenizer,
                        str(payload["text"]),
                        Path(payload["out_dir"]),
                        int(payload["max_seq_len"]),
                        int(payload.get("scale_bits", 16)),
                    )
                    write_text_atomic(done_path, f"elapsed_s={time.perf_counter() - t0:.9f}\n")
                except Exception as exc:  # noqa: BLE001
                    write_text_atomic(error_path, repr(exc))
            time.sleep(0.05)
    finally:
        with suppress(OSError):
            ready_path.unlink(missing_ok=True)
        with suppress(OSError):
            heartbeat_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
