"""
BERT 三路径推理集成测试。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BERT_DIR = os.path.join(ROOT, "bert-base-uncased")
SKIP_NO_BERT = not os.path.isdir(BERT_DIR)

SMOKE_TEXTS = [
    "This movie is wonderful and heartwarming.",
    "A complete waste of time, absolutely terrible.",
    "The acting was okay but the plot was boring.",
    "I loved every minute of it!",
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if SKIP_NO_BERT:
        pytest.skip("bert-base-uncased 未下载")
    from transformer.bert_weight_loader import load_classification_model, load_tokenizer

    model = load_classification_model("cpu")
    tokenizer = load_tokenizer()
    return model, tokenizer


@pytest.mark.skipif(SKIP_NO_BERT, reason="no bert weights")
def test_plaintext_smoke(model_and_tokenizer):
    from transformer.plaintext_bert import classify_plaintext_hf

    model, tokenizer = model_and_tokenizer
    for text in SMOKE_TEXTS:
        label, probs = classify_plaintext_hf(model, tokenizer, text, "cpu", max_seq_len=16)
        assert label in ("negative", "positive")
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-4


@pytest.mark.skipif(SKIP_NO_BERT, reason="no bert weights")
def test_layer0_align(model_and_tokenizer):
    from transformer.bert_weight_loader import NUM_HEADS, extract_all_layer_weights
    from transformer.mcu_bert_stack import encoder_layer_forward, extended_attention_mask
    from transformer.plaintext_bert import embed_inputs, tokenize

    model, tokenizer = model_and_tokenizer
    inputs = tokenize(tokenizer, SMOKE_TEXTS[0], 16, "cpu")
    lw = extract_all_layer_weights(model)
    hidden = embed_inputs(model, inputs)
    attn = extended_attention_mask(inputs["attention_mask"], hidden.dtype)
    plain = encoder_layer_forward(hidden, lw[0], NUM_HEADS, attn, "plain")
    mcu = encoder_layer_forward(hidden, lw[0], NUM_HEADS, attn, "mcu_rust")
    err = (plain - mcu).abs().max().item()
    assert err < 0.2, f"layer0 MCU 对齐误差过大: {err}"


@pytest.mark.skipif(SKIP_NO_BERT, reason="no bert weights")
def test_crypten_2quad_degrades(model_and_tokenizer):
    import torch
    from transformer.mcu_nonlinear import two_quad

    scores = torch.randn(1, 12, 8, 8)
    soft = torch.softmax(scores, dim=-1)
    approx = two_quad(scores)
    diff = (soft - approx).abs().max().item()
    assert diff > 0.01


@pytest.mark.skipif(SKIP_NO_BERT, reason="no bert weights")
def test_three_way_labels(model_and_tokenizer):
    from transformer.mcu_bert_crypten_full import classify_crypten_full
    from transformer.mcu_bert_rust import classify_mcu_rust_full
    from transformer.plaintext_bert import classify_plaintext_hf

    model, tokenizer = model_and_tokenizer
    dataset = [
        ("This movie is wonderful.", 1),
        ("A complete waste of time.", 0),
        ("I loved every minute of it!", 1),
        ("Terrible and boring film.", 0),
    ]
    names = ["negative", "positive"]
    acc_ct, acc_mcu = 0, 0
    for text, lab in dataset:
        pl, _ = classify_plaintext_hf(model, tokenizer, text, "cpu", 16)
        cl, _, _ = classify_crypten_full(model, tokenizer, text, "cpu", 16)
        ml, _, _ = classify_mcu_rust_full(model, tokenizer, text, "cpu", 16)
        if names.index(cl) == lab:
            acc_ct += 1
        if names.index(ml) == lab:
            acc_mcu += 1
    assert acc_mcu >= acc_ct, f"mcu={acc_mcu} crypten={acc_ct}"


@pytest.mark.skipif(SKIP_NO_BERT, reason="no bert weights")
def test_api_semantic():
    sys.path.insert(0, os.path.join(ROOT, "dashboard", "backend"))
    import bert_inference
    bert_inference._engine = None

    from fastapi.testclient import TestClient

    sys.path.insert(0, os.path.join(ROOT, "dashboard", "backend"))
    from main import app

    client = TestClient(app)
    body = {"text": "This movie is wonderful.", "mode": "plaintext", "max_seq_len": 16}
    for mode in ("plaintext", "crypten", "mcu_rust"):
        body["mode"] = mode
        r = client.post("/api/infer/semantic", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("success"), data.get("error", data)
        assert data["label"] in ("negative", "positive")


def test_rust_verify_endpoint():
    from fastapi.testclient import TestClient

    sys.path.insert(0, os.path.join(ROOT, "dashboard", "backend"))
    from main import app

    client = TestClient(app)
    r = client.get("/api/rust/verify")
    assert r.status_code == 200
    assert "available" in r.json()
