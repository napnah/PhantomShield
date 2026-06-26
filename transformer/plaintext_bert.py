"""
明文 BERT 前向：Embedding → 12 层 Encoder → Pooler → Classifier。
供基准准确率与逐层对齐使用。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

from transformer.bert_weight_loader import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    extract_all_layer_weights,
    extract_classifier_weights,
)
from transformer.mcu_bert_stack import encoder_layer_forward, encoder_stack_forward, extended_attention_mask


@torch.no_grad()
def tokenize(
    tokenizer: BertTokenizer,
    text: str,
    max_seq_len: int = 128,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def embed_inputs(model: BertForSequenceClassification, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.bert.embeddings(input_ids=inputs["input_ids"], token_type_ids=None)


@torch.no_grad()
def pooler_forward(model: BertForSequenceClassification, hidden: torch.Tensor) -> torch.Tensor:
    return model.bert.pooler(hidden)


@torch.no_grad()
def classify_logits(model: BertForSequenceClassification, cls: torch.Tensor) -> torch.Tensor:
    return model.classifier(cls)


@torch.no_grad()
def forward_encoder_layer_at(
    model: BertForSequenceClassification,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
    nonlinear: str = "plain",
) -> torch.Tensor:
    """运行 embedding + layers[0:layer_idx+1]，返回该层输出。"""
    layer_weights = extract_all_layer_weights(model)
    hidden = embed_inputs(model, inputs)
    attn_ext = extended_attention_mask(inputs["attention_mask"], hidden.dtype)
    return encoder_stack_forward(
        hidden, layer_weights, NUM_HEADS, attn_ext, nonlinear=nonlinear, max_layers=layer_idx + 1
    )


@torch.no_grad()
def forward_plaintext_encoder(
    model: BertForSequenceClassification,
    inputs: dict[str, torch.Tensor],
    nonlinear: str = "plain",
    max_layers: int | None = None,
) -> torch.Tensor:
    layer_weights = extract_all_layer_weights(model)
    hidden = embed_inputs(model, inputs)
    attn_ext = extended_attention_mask(inputs["attention_mask"], hidden.dtype)
    return encoder_stack_forward(
        hidden, layer_weights, NUM_HEADS, attn_ext, nonlinear=nonlinear, max_layers=max_layers
    )


@torch.no_grad()
def classify_plaintext_full(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    device: str = "cpu",
    max_seq_len: int = 128,
    nonlinear: str = "plain",
) -> tuple[str, list[float], float]:
    """完整明文路径：12 层 Encoder + pooler + classifier。"""
    inputs = tokenize(tokenizer, text, max_seq_len, device)
    hidden = forward_plaintext_encoder(model, inputs, nonlinear=nonlinear)
    cls = pooler_forward(model, hidden)
    logits = classify_logits(model, cls)
    probs = F.softmax(logits, dim=-1)[0].cpu()
    labels = ["negative", "positive"]
    return labels[int(probs.argmax())], probs.tolist(), float(probs.max())


@torch.no_grad()
def classify_plaintext_hf(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    device: str = "cpu",
    max_seq_len: int = 128,
) -> tuple[str, list[float]]:
    """标准 HuggingFace 全量前向（基准）。"""
    inputs = tokenize(tokenizer, text, max_seq_len, device)
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu()
    labels = ["negative", "positive"]
    return labels[int(probs.argmax())], probs.tolist()


LABELS = ["negative", "positive"]
