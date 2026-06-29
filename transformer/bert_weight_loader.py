"""
从 HuggingFace BERT 提取每层 Encoder 权重为统一 dict 格式。
"""
import os
from typing import Any

import torch
from transformers import BertForSequenceClassification, BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "bert-base-uncased")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "bert-sst2")

HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
NUM_LAYERS = 12
FFN_SIZE = 3072


def _t(w: torch.Tensor) -> torch.Tensor:
    """HF Linear weight (out, in) -> matmul 用 (in, out)。"""
    return w.detach().float().T.contiguous()


def extract_layer_weights(model: BertForSequenceClassification, layer_idx: int) -> dict[str, torch.Tensor]:
    layer = model.bert.encoder.layer[layer_idx]
    ln1 = layer.attention.output.LayerNorm
    ln2 = layer.output.LayerNorm
    return {
        "Wq": _t(layer.attention.self.query.weight),
        "b_q": layer.attention.self.query.bias.detach().float().clone(),
        "Wk": _t(layer.attention.self.key.weight),
        "b_k": layer.attention.self.key.bias.detach().float().clone(),
        "Wv": _t(layer.attention.self.value.weight),
        "b_v": layer.attention.self.value.bias.detach().float().clone(),
        "Wo": _t(layer.attention.output.dense.weight),
        "b_o": layer.attention.output.dense.bias.detach().float().clone(),
        "W1": _t(layer.intermediate.dense.weight),
        "b1": layer.intermediate.dense.bias.detach().float().clone(),
        "W2": _t(layer.output.dense.weight),
        "b2": layer.output.dense.bias.detach().float().clone(),
        "ln1_g": ln1.weight.detach().float().clone(),
        "ln1_b": ln1.bias.detach().float().clone(),
        "ln2_g": ln2.weight.detach().float().clone(),
        "ln2_b": ln2.bias.detach().float().clone(),
    }


def extract_all_layer_weights(model: BertForSequenceClassification) -> list[dict[str, torch.Tensor]]:
    n = len(model.bert.encoder.layer)
    return [extract_layer_weights(model, i) for i in range(n)]


def extract_classifier_weights(model: BertForSequenceClassification) -> tuple[torch.Tensor, torch.Tensor]:
    return _t(model.classifier.weight), model.classifier.bias.detach().float().clone()


def load_tokenizer() -> BertTokenizer:
    path = CKPT_DIR if os.path.isdir(CKPT_DIR) else MODEL_DIR
    return BertTokenizer.from_pretrained(path)


def load_classification_model(device: str = "cpu") -> BertForSequenceClassification:
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"未找到 {MODEL_DIR}，请先运行 scripts/download_models.py")
    path = CKPT_DIR if os.path.isfile(os.path.join(CKPT_DIR, "config.json")) else MODEL_DIR
    model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
    return model.to(device).eval()


def model_paths() -> dict[str, Any]:
    return {
        "model_dir": MODEL_DIR,
        "ckpt_dir": CKPT_DIR,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "num_layers": NUM_LAYERS,
        "ffn_size": FFN_SIZE,
    }
