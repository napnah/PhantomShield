"""
BERT 三路径推理服务（Dashboard 后端单例）。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Literal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from transformer.bert_weight_loader import load_classification_model, load_tokenizer
from transformer.mcu_bert_crypten_full import classify_crypten_full
from transformer.mcu_bert_rust import classify_mcu_rust_full
from transformer.plaintext_bert import classify_plaintext_hf

Mode = Literal["plaintext", "crypten", "mcu_rust"]

_engine = None


class BertInferenceEngine:
    def __init__(self, device: str | None = None):
        # 三路径（CrypTen / mcu_rust）在 CPU 上稳定；统一 CPU 推理
        self.device = device or "cpu"
        self.model = load_classification_model(self.device)
        self.tokenizer = load_tokenizer()
        self.layer_count = 12

    def classify(
        self,
        text: str,
        mode: Mode = "plaintext",
        max_seq_len: int = 32,
    ) -> dict:
        t0 = time.time()
        if mode == "plaintext":
            label, probs = classify_plaintext_hf(
                self.model, self.tokenizer, text, self.device, max_seq_len
            )
            method = "Plaintext BERT (HuggingFace 全量前向)"
        elif mode == "crypten":
            label, probs, _ = classify_crypten_full(
                self.model, self.tokenizer, text, self.device, max_seq_len
            )
            method = "CrypTen 12L Encoder + 2Quad Softmax + Quad FFN + 2Quad 分类头"
        elif mode == "mcu_rust":
            label, probs, _ = classify_mcu_rust_full(
                self.model, self.tokenizer, text, self.device, max_seq_len
            )
            method = "MCU-Rust 12L Encoder + Π_softmax (精确) + Plain GeLU + 分类头"
        else:
            raise ValueError(f"unknown mode: {mode}")

        elapsed = time.time() - t0
        conf = float(max(probs))
        dist = [
            {"label": "negative" if i == 0 else "positive", "prob": round(p * 100, 1)}
            for i, p in enumerate(probs)
        ]
        dist.sort(key=lambda x: -x["prob"])

        return {
            "success": True,
            "mode": mode,
            "method": method,
            "label": label,
            "confidence": round(conf * 100, 1),
            "distribution": dist,
            "probabilities": probs,
            "elapsed_seconds": round(elapsed, 3),
            "layer_count": self.layer_count,
            "input": text,
        }


def get_engine() -> BertInferenceEngine:
    global _engine
    if _engine is None:
        _engine = BertInferenceEngine(device="cpu")
    return _engine
