"""
共享 BERT Encoder 层前向：多头注意力 + FFN + LayerNorm。
支持 plain / two_quad / mcu_rust 三种非线性模式。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from transformer.mcu_nonlinear import gelu_approx, mcu_rust_gelu, mcu_rust_softmax_rows, two_quad


def extended_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """attention_mask (B,S) 1=有效 -> (B,1,1,S) 加性 mask。"""
    ext = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - ext) * -10000.0


def layer_norm_plain(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps=eps)


def attention_forward(
    x: torch.Tensor,
    weights: dict,
    num_heads: int,
    attn_mask_ext: torch.Tensor | None = None,
    nonlinear: str = "plain",
) -> torch.Tensor:
    """
    x: (B, S, H)
    nonlinear: plain | two_quad | mcu_rust
    """
    b, s, h = x.shape
    head_dim = h // num_heads
    scale = head_dim ** 0.5

    q = (x @ weights["Wq"] + weights["b_q"]).view(b, s, num_heads, head_dim).transpose(1, 2)
    k = (x @ weights["Wk"] + weights["b_k"]).view(b, s, num_heads, head_dim).transpose(1, 2)
    v = (x @ weights["Wv"] + weights["b_v"]).view(b, s, num_heads, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / scale
    if attn_mask_ext is not None:
        scores = scores + attn_mask_ext

    if nonlinear == "plain":
        probs = torch.softmax(scores, dim=-1)
    elif nonlinear == "two_quad":
        probs = two_quad(scores)
    elif nonlinear == "mcu_rust":
        probs = mcu_rust_softmax_rows(scores)
    else:
        raise ValueError(f"unknown nonlinear: {nonlinear}")

    ctx = probs @ v
    ctx = ctx.transpose(1, 2).contiguous().view(b, s, h)
    out = ctx @ weights["Wo"] + weights["b_o"]
    return out


def ffn_forward(
    x: torch.Tensor,
    weights: dict,
    nonlinear: str = "plain",
) -> torch.Tensor:
    h = x @ weights["W1"] + weights["b1"]
    if nonlinear == "plain":
        h = F.gelu(h)
    elif nonlinear == "two_quad":
        h = gelu_approx(h)
    elif nonlinear == "mcu_rust":
        h = mcu_rust_gelu(h)
    else:
        raise ValueError(f"unknown nonlinear: {nonlinear}")
    return h @ weights["W2"] + weights["b2"]


def encoder_layer_forward(
    x: torch.Tensor,
    weights: dict,
    num_heads: int,
    attn_mask_ext: torch.Tensor | None = None,
    nonlinear: str = "plain",
) -> torch.Tensor:
    """单层 Encoder：Attn + LN + FFN + LN。"""
    attn_nl = nonlinear
    ffn_nl = nonlinear
    if nonlinear == "mcu_rust":
        attn_nl = "mcu_rust"
        ffn_nl = "plain"  # 精确 GeLU；与 CrypTen gelu_approx 区分

    attn_out = attention_forward(x, weights, num_heads, attn_mask_ext, nonlinear=attn_nl)
    x = layer_norm_plain(x + attn_out, weights["ln1_g"], weights["ln1_b"])

    ffn_out = ffn_forward(x, weights, nonlinear=ffn_nl)
    x = layer_norm_plain(x + ffn_out, weights["ln2_g"], weights["ln2_b"])
    return x


def encoder_stack_forward(
    hidden: torch.Tensor,
    layer_weights: list[dict],
    num_heads: int,
    attn_mask_ext: torch.Tensor | None = None,
    nonlinear: str = "plain",
    max_layers: int | None = None,
) -> torch.Tensor:
    n = max_layers if max_layers is not None else len(layer_weights)
    x = hidden
    for i in range(n):
        x = encoder_layer_forward(x, layer_weights[i], num_heads, attn_mask_ext, nonlinear=nonlinear)
    return x
