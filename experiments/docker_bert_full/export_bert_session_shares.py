from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformer.bert_weight_loader import (  # noqa: E402
    extract_all_layer_weights,
    extract_classifier_weights,
    load_classification_model,
    load_tokenizer,
    model_paths,
)
from transformer.plaintext_bert import embed_inputs  # noqa: E402


DEFAULT_TEXT = "This movie is wonderful and heartwarming."
DEFAULT_SCALE_BITS = 16
UINT64_MOD = 1 << 64


def splitmix64_stream(seed: int, length: int) -> np.ndarray:
    values = np.empty(length, dtype=np.uint64)
    state = np.uint64(seed)
    inc = np.uint64(0x9E3779B97F4A7C15)
    mul1 = np.uint64(0xBF58476D1CE4E5B9)
    mul2 = np.uint64(0x94D049BB133111EB)
    with np.errstate(over="ignore"):
        for i in range(length):
            state = state + inc
            z = state
            z = (z ^ (z >> np.uint64(30))) * mul1
            z = (z ^ (z >> np.uint64(27))) * mul2
            values[i] = z ^ (z >> np.uint64(31))
    return values


def tensor_to_ring_u64(tensor: torch.Tensor, scale_bits: int) -> np.ndarray:
    scale = float(1 << scale_bits)
    fixed = torch.round(tensor.detach().cpu().float().reshape(-1) * scale).to(torch.int64).numpy()
    return fixed.astype(np.uint64)


def write_shared_tensor(
    name: str,
    tensor: torch.Tensor,
    p0_dir: Path,
    p1_dir: Path,
    rows: list[dict],
    seed: int,
    scale_bits: int,
) -> None:
    values = tensor_to_ring_u64(tensor, scale_bits)
    share0 = splitmix64_stream(seed, values.size)
    share1 = values - share0
    file_name = f"{name}.bin"
    share0.tofile(p0_dir / file_name)
    share1.tofile(p1_dir / file_name)
    rows.append(
        {
            "name": name,
            "file": file_name,
            "shape": "x".join(str(x) for x in tensor.shape),
            "numel": int(values.size),
            "scale_bits": scale_bits,
            "seed": seed,
        }
    )


def write_public_u64_tensor(
    name: str,
    tensor: torch.Tensor,
    p0_dir: Path,
    p1_dir: Path,
    rows: list[dict],
) -> None:
    values = tensor.detach().cpu().reshape(-1).to(torch.int64).numpy().astype(np.uint64)
    file_name = f"{name}.bin"
    values.tofile(p0_dir / file_name)
    values.tofile(p1_dir / file_name)
    rows.append(
        {
            "name": name,
            "file": file_name,
            "shape": "x".join(str(x) for x in tensor.shape),
            "numel": int(values.size),
            "scale_bits": 0,
            "seed": "public",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--samples-json", help="JSON file with a list of texts or {text,label} objects.")
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=DEFAULT_SCALE_BITS)
    parser.add_argument("--device", default="cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static-only", action="store_true", help="Export model/checkpoint shares only.")
    group.add_argument("--input-only", action="store_true", help="Export input hidden/mask shares only.")
    args = parser.parse_args()

    model = load_classification_model(args.device)
    labels: list[int | None] = []
    if args.samples_json:
        raw_samples = json.loads(Path(args.samples_json).read_text(encoding="utf-8-sig"))
        texts = [sample["text"] if isinstance(sample, dict) else str(sample) for sample in raw_samples]
        labels = [sample.get("label") if isinstance(sample, dict) else None for sample in raw_samples]
    else:
        texts = [args.text]
        labels = [None]
    hidden = None
    inputs = None
    if not args.static_only:
        tokenizer = load_tokenizer()
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )
        inputs = {k: v.to(args.device) for k, v in enc.items()}
        hidden = embed_inputs(model, inputs)
    max_layers = min(args.layers, len(model.bert.encoder.layer))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_bert_session_shares"
    p0_dir = out / "p0"
    p1_dir = out / "p1"
    p0_dir.mkdir(parents=True, exist_ok=True)
    p1_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if not args.static_only:
        assert hidden is not None and inputs is not None
        write_shared_tensor("hidden", hidden.reshape(-1, hidden.shape[-1]), p0_dir, p1_dir, rows, 0xBEE70001, args.scale_bits)
        write_public_u64_tensor("attention_mask", inputs["attention_mask"], p0_dir, p1_dir, rows)

    seed = 0xC0010000
    wanted = [
        "Wq",
        "b_q",
        "Wk",
        "b_k",
        "Wv",
        "b_v",
        "Wo",
        "W1",
        "W2",
        "b_o",
        "b1",
        "b2",
        "ln1_g",
        "ln1_b",
        "ln2_g",
        "ln2_b",
    ]
    if not args.input_only:
        all_weights = extract_all_layer_weights(model)
        for layer_idx in range(max_layers):
            for name in wanted:
                write_shared_tensor(
                    f"layer_{layer_idx:02d}_{name}",
                    all_weights[layer_idx][name],
                    p0_dir,
                    p1_dir,
                    rows,
                    seed,
                    args.scale_bits,
                )
                seed += 1
        write_shared_tensor(
            "pooler_W",
            model.bert.pooler.dense.weight.detach().float().T.contiguous(),
            p0_dir,
            p1_dir,
            rows,
            seed,
            args.scale_bits,
        )
        seed += 1
        write_shared_tensor(
            "pooler_b",
            model.bert.pooler.dense.bias.detach().float().clone(),
            p0_dir,
            p1_dir,
            rows,
            seed,
            args.scale_bits,
        )
        seed += 1
        classifier_w, classifier_b = extract_classifier_weights(model)
        write_shared_tensor("classifier_W", classifier_w, p0_dir, p1_dir, rows, seed, args.scale_bits)
        seed += 1
        write_shared_tensor("classifier_b", classifier_b, p0_dir, p1_dir, rows, seed, args.scale_bits)

    write_csv(out / "tensor_manifest.csv", rows)
    paths = model_paths()
    manifest = {
        "purpose": "Real BERT embedding and checkpoint weight shares for MCU bert_session real_io mode.",
        "export_mode": "static_only" if args.static_only else "input_only" if args.input_only else "full",
        "text": args.text if len(texts) == 1 else None,
        "texts": texts,
        "labels": labels,
        "max_seq_len": args.max_seq_len,
        "scale_bits": args.scale_bits,
        "layers": max_layers,
        "shape": {
            "batch": int(hidden.shape[0]) if hidden is not None else len(texts),
            "seq": int(hidden.shape[1]) if hidden is not None else args.max_seq_len,
            "hidden": int(hidden.shape[2]) if hidden is not None else paths["hidden_size"],
            "heads": paths["num_heads"],
            "head_dim": paths["head_dim"],
            "ffn": paths["ffn_size"],
        },
        "implemented_tensors": [row["name"] for row in rows],
        "not_yet_exported": [],
        "numerical_boundary": [
            "Matmul inputs are fixed-point ring shares.",
            "Secure rescale/truncation after matmul is not wired into bert_session yet.",
            "Current real_io mode validates real tensor IO and protocol consumption, not full BERT numerical equivalence.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[bert-shares] wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
