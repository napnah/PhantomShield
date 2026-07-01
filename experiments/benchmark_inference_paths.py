"""
三路径 BERT 推理基准：明文 / CrypTen(12L+2Quad) / mcu_rust(12L)。

输出 results/inference_benchmark.json，验收 acc_mcu_rust >= acc_crypten。

运行（项目根目录）：
    python experiments/benchmark_inference_paths.py
    python experiments/benchmark_inference_paths.py --smoke   # 4 条快速
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer.bert_weight_loader import load_classification_model, load_tokenizer
from transformer.mcu_bert_crypten_full import classify_crypten_full
from transformer.mcu_bert_rust import classify_mcu_rust_full
from transformer.plaintext_bert import classify_plaintext_hf

RESULTS_PATH = os.path.join(ROOT, "results", "inference_benchmark.json")

SMOKE_SET = [
    ("This movie is wonderful and heartwarming.", 1),
    ("A complete waste of time, absolutely terrible.", 0),
    ("The acting was okay but the plot was boring.", 0),
    ("I loved every minute of it!", 1),
]

VAL_SET = [
    ("This movie is wonderful and heartwarming.", 1),
    ("I loved every minute of it!", 1),
    ("Brilliant acting and a great story.", 1),
    ("A delightful film with amazing performances.", 1),
    ("A complete waste of time, absolutely terrible.", 0),
    ("Boring, predictable, and poorly acted.", 0),
    ("I hated this film from start to finish.", 0),
    ("The worst movie I have seen this year.", 0),
    ("Painfully slow and utterly pointless.", 0),
    ("Nothing made sense and it was dull.", 0),
]


def _acc(preds: list[int], labels: list[int]) -> float:
    if not preds:
        return 0.0
    return sum(p == l for p, l in zip(preds, labels)) / len(preds)


def _run_mode(name, fn, model, tokenizer, dataset, device, max_seq_len):
    preds, latencies = [], []
    label_names = ["negative", "positive"]
    for text, lab in dataset:
        t0 = time.time()
        out = fn(model, tokenizer, text, device, max_seq_len)
        latencies.append(time.time() - t0)
        pred_label = out[0]
        preds.append(label_names.index(pred_label))
    return {
        "accuracy": round(_acc(preds, [l for _, l in dataset]), 4),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3),
        "n_samples": len(dataset),
    }


def run_benchmark(smoke: bool = False, max_seq_len: int = 32) -> dict:
    import torch
    # CrypTen 全 12 层在 CPU 上运行更稳定
    device = "cpu"
    dataset = SMOKE_SET if smoke else VAL_SET
    print(f"设备: {device} | 样本: {len(dataset)} | max_seq_len={max_seq_len}")

    model = load_classification_model(device)
    tokenizer = load_tokenizer()

    results = {
        "dataset": "smoke_4" if smoke else "mini_sst2_val",
        "max_seq_len": max_seq_len,
        "layer_count": 12,
    }

    print("\n[1/3] 明文 BERT 基准...")
    results["plaintext"] = _run_mode(
        "plain", classify_plaintext_hf, model, tokenizer, dataset, device, max_seq_len
    )
    print(f"  acc={results['plaintext']['accuracy']} latency={results['plaintext']['avg_latency_s']}s")

    print("\n[2/3] CrypTen 12L + 2Quad...")
    results["crypten"] = _run_mode(
        "crypten", classify_crypten_full, model, tokenizer, dataset, device, max_seq_len
    )
    print(f"  acc={results['crypten']['accuracy']} latency={results['crypten']['avg_latency_s']}s")

    print("\n[3/3] mcu_rust 12L...")
    results["mcu_rust"] = _run_mode(
        "mcu_rust", classify_mcu_rust_full, model, tokenizer, dataset, device, max_seq_len
    )
    print(f"  acc={results['mcu_rust']['accuracy']} latency={results['mcu_rust']['avg_latency_s']}s")

    acc_p = results["plaintext"]["accuracy"]
    acc_c = results["crypten"]["accuracy"]
    acc_m = results["mcu_rust"]["accuracy"]
    results["pass"] = {
        "mcu_ge_crypten": acc_m >= acc_c,
        "mcu_near_plain": acc_m >= acc_p - 0.15,
        "plaintext_baseline": acc_p,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已写入 {RESULTS_PATH}")
    print(f"验收 mcu>=crypten: {results['pass']['mcu_ge_crypten']}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="仅 4 条样本")
    parser.add_argument("--max-seq-len", type=int, default=32)
    args = parser.parse_args()
    results = run_benchmark(smoke=args.smoke, max_seq_len=args.max_seq_len)
    ok = results["pass"]["mcu_ge_crypten"]
    if not ok:
        print("[FAIL] mcu_rust 准确率未超过 CrypTen")
        return 1
    print("[OK] 三路径基准验收通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
