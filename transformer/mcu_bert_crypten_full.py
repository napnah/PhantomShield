"""
12 层 BERT Encoder 密文前向（CrypTen）+ 2Quad 注意力 Softmax。
Embedding 明文；LayerNorm 方差揭示（与 mcu_bert_crypten 一致）。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from crypten_compat import patch_crypten_inprocess

from transformer.bert_weight_loader import NUM_HEADS, extract_all_layer_weights, extract_classifier_weights
from transformer.mcu_nonlinear import gelu_approx, two_quad


def quad_activation(x: torch.Tensor) -> torch.Tensor:
    """SecFormer Quad 激活（替换 GeLU，精度更低）。"""
    return 0.125 * x ** 2 + 0.25 * x + 0.5
from transformer.plaintext_bert import embed_inputs, pooler_forward, tokenize
from transformers import BertForSequenceClassification, BertTokenizer

def _ensure_crypten():
    import crypten

    patch_crypten_inprocess()
    crypten.init_thread(0, 1)
    return crypten


def _enc(x: torch.Tensor):
    crypten = _ensure_crypten()
    return crypten.cryptensor(x)


def _layernorm_crypten(x_enc, gamma, beta, eps=1e-5):
    mean = x_enc.mean(dim=-1, keepdim=True)
    diff = x_enc - mean
    var = (diff * diff).mean(dim=-1, keepdim=True)
    var_plain = var.get_plain_text()
    inv_std = 1.0 / torch.sqrt(var_plain + eps)
    return diff * inv_std * gamma + beta


def _attention_crypten(x_enc, weights, num_heads, attn_mask_ext):
    b, s, h = x_enc.size()
    head_dim = h // num_heads
    scale = head_dim ** 0.5

    wq, wk, wv, wo = weights["Wq"], weights["Wk"], weights["Wv"], weights["Wo"]
    bo = weights["b_o"]

    q = x_enc.matmul(wq).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = x_enc.matmul(wk).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = x_enc.matmul(wv).view(b, s, num_heads, head_dim).transpose(1, 2)

    scores = q.matmul(k.transpose(2, 3)) / scale
    if attn_mask_ext is not None:
        scores = scores + attn_mask_ext

    # 2Quad 在明文 scores 上计算后加密（CrypTen 张量上二次运算等价）
    scores_plain = scores.get_plain_text()
    probs_plain = two_quad(scores_plain)
    probs = _enc(probs_plain)

    ctx = probs.matmul(v)
    ctx_plain = ctx.get_plain_text().transpose(1, 2).reshape(b, s, h)
    return _enc(ctx_plain).matmul(wo) + bo


def _ffn_crypten(x_enc, weights):
    h = x_enc.matmul(weights["W1"]) + weights["b1"]
    h_plain = h.get_plain_text()
    h_act = quad_activation(h_plain)
    h = _enc(h_act)
    return h.matmul(weights["W2"]) + weights["b2"]


def _encoder_layer_crypten(x_enc, weights, num_heads, attn_mask_ext):
    attn = _attention_crypten(x_enc, weights, num_heads, attn_mask_ext)
    x_enc = _layernorm_crypten(x_enc + attn, weights["ln1_g"], weights["ln1_b"])
    ffn = _ffn_crypten(x_enc, weights)
    x_enc = _layernorm_crypten(x_enc + ffn, weights["ln2_g"], weights["ln2_b"])
    return x_enc


def forward_crypten_encoder(
    model: BertForSequenceClassification,
    inputs: dict[str, torch.Tensor],
    max_layers: int = 12,
) -> torch.Tensor:
    from transformer.mcu_bert_stack import extended_attention_mask

    layer_weights = [{k: v.cpu() for k, v in w.items()} for w in extract_all_layer_weights(model)]
    hidden = embed_inputs(model, inputs).detach().cpu()
    attn_ext = extended_attention_mask(inputs["attention_mask"].cpu(), hidden.dtype)

    x_enc = _enc(hidden)
    for i in range(max_layers):
        x_enc = _encoder_layer_crypten(x_enc, layer_weights[i], NUM_HEADS, attn_ext)
    return x_enc.get_plain_text()


@torch.no_grad()
def classify_crypten_full(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    device: str = "cpu",
    max_seq_len: int = 32,
    max_layers: int = 12,
) -> tuple[str, list[float], float]:
    """12 层 CrypTen Encoder（2Quad）+ 明文 pooler + CrypTen 分类头。"""
    _ensure_crypten()
    import crypten

    inputs = tokenize(tokenizer, text, max_seq_len, device)
    hidden_plain = forward_crypten_encoder(model, inputs, max_layers=max_layers)
    cls = pooler_forward(model, hidden_plain.to(device))

    w, b = extract_classifier_weights(model)
    cls_enc = crypten.cryptensor(cls.cpu())
    logits_enc = cls_enc.matmul(crypten.cryptensor(w.cpu())) + crypten.cryptensor(b.cpu())
    logits_plain = logits_enc.get_plain_text()
    probs = two_quad(logits_plain)[0]
    labels = ["negative", "positive"]
    pred = int(probs.argmax())
    return labels[pred], probs.tolist(), float(probs.max())
