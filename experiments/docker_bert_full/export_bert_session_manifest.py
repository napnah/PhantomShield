from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformer.bert_weight_loader import (  # noqa: E402
    extract_all_layer_weights,
    extract_classifier_weights,
    load_classification_model,
    load_tokenizer,
    model_paths,
)
from transformer.plaintext_bert import embed_inputs, tokenize  # noqa: E402


DEFAULT_TEXT = "This movie is wonderful and heartwarming."


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
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / f"{timestamp}_bert_session_manifest"
    out.mkdir(parents=True, exist_ok=True)

    model = load_classification_model(args.device)
    tokenizer = load_tokenizer()
    enc = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
    )
    inputs = {k: v.to(args.device) for k, v in enc.items()}
    hidden = embed_inputs(model, inputs)
    layer_weights = extract_all_layer_weights(model)
    classifier_w, classifier_b = extract_classifier_weights(model)
    paths = model_paths()

    rows = []
    for layer_idx, weights in enumerate(layer_weights):
        for name, tensor in weights.items():
            rows.append(
                {
                    "layer": layer_idx,
                    "name": name,
                    "shape": "x".join(str(x) for x in tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
            )
    write_csv(out / "layer_weight_shapes.csv", rows)
    write_csv(
        out / "classifier_shapes.csv",
        [
            {
                "name": "classifier_weight_transposed",
                "shape": "x".join(str(x) for x in classifier_w.shape),
                "dtype": str(classifier_w.dtype),
                "numel": classifier_w.numel(),
            },
            {
                "name": "classifier_bias",
                "shape": "x".join(str(x) for x in classifier_b.shape),
                "dtype": str(classifier_b.dtype),
                "numel": classifier_b.numel(),
            },
        ],
    )

    manifest = {
        "purpose": "Interface manifest for wiring real BERT tensors into MCU bert_session.",
        "model": paths,
        "input": {
            "text": args.text,
            "max_seq_len": args.max_seq_len,
            "input_ids_shape": list(inputs["input_ids"].shape),
            "attention_mask_shape": list(inputs["attention_mask"].shape),
            "embedding_shape": list(hidden.shape),
            "effective_seq_len": int(inputs["attention_mask"].sum().item()),
        },
        "bert_session_shape": {
            "batch": int(hidden.shape[0]),
            "seq": int(hidden.shape[1]),
            "hidden": int(hidden.shape[2]),
            "heads": paths["num_heads"],
            "head_dim": paths["head_dim"],
            "ffn": paths["ffn_size"],
            "layers": paths["num_layers"],
        },
        "next_required_mapping": [
            "Feed embedding shares as the session input state instead of synthetic shares.",
            "Use Wq/Wk/Wv/Wo/W1/W2 tensors from layer_weight_shapes.csv as model-side shares.",
            "Add secure residual add and LayerNorm after attention and FFN.",
            "Propagate output shares between modules instead of regenerating synthetic inputs per module.",
            "Add pooler and classifier matmul at the end of the session.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[bert-manifest] wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
