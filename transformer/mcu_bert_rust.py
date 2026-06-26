"""
12 层 BERT Encoder：线性明文 matmul + mcu_rust 精确 Π_softmax / Π_gelu。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

from transformer.bert_weight_loader import NUM_HEADS, extract_classifier_weights
from transformer.mcu_bert_stack import encoder_stack_forward, extended_attention_mask
from transformer.plaintext_bert import embed_inputs, pooler_forward, tokenize


@torch.no_grad()
def forward_mcu_rust_encoder(
    model: BertForSequenceClassification,
    inputs: dict[str, torch.Tensor],
    max_layers: int = 12,
) -> torch.Tensor:
    from transformer.bert_weight_loader import extract_all_layer_weights

    layer_weights = extract_all_layer_weights(model)
    hidden = embed_inputs(model, inputs)
    attn_ext = extended_attention_mask(inputs["attention_mask"], hidden.dtype)
    return encoder_stack_forward(
        hidden, layer_weights, NUM_HEADS, attn_ext, nonlinear="mcu_rust", max_layers=max_layers
    )


@torch.no_grad()
def classify_mcu_rust_full(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    device: str = "cpu",
    max_seq_len: int = 32,
    max_layers: int = 12,
) -> tuple[str, list[float], float]:
    """12 层 mcu_rust 非线性 Encoder + pooler + mcu_rust softmax 分类头。"""
    inputs = tokenize(tokenizer, text, max_seq_len, device)
    hidden = forward_mcu_rust_encoder(model, inputs, max_layers=max_layers)
    cls = pooler_forward(model, hidden)

    w, b = extract_classifier_weights(model)
    logits = cls @ w + b
    probs_t = F.softmax(logits, dim=-1).squeeze(0)
    labels = ["negative", "positive"]
    pred = int(probs_t.argmax())
    return labels[pred], probs_t.tolist(), float(probs_t.max())
