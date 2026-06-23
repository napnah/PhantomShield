"""
CrypTen + BERT 语义分类演示（SST-2 风格情感二分类）

流程：
  1. 在内置英文情感样本上快速微调 BERT-Base 分类头（无需下载 GLUE）
  2. 明文推理：标准 HuggingFace forward
  3. 密文推理：P0 本地 BERT 编码器 → [CLS] 向量；分类头 CrypTen 密文 matmul + softmax

说明：完整 12 层 BERT 密文前向在 CrypTen 下过慢；本脚本对分类头走 CrypTen 2PC，
验证密文 softmax 分类与明文一致。

运行（项目根目录）：
    python experiments/crypten_bert_classify.py
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from crypten_compat import patch_crypten_inprocess

MODEL_DIR = os.path.join(ROOT, "bert-base-uncased")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "bert-sst2")

# 内置迷你情感语料（label: 0=negative, 1=positive）
MINI_CORPUS = [
    ("This movie is wonderful and heartwarming.", 1),
    ("I loved every minute of it!", 1),
    ("Brilliant acting and a great story.", 1),
    ("Absolutely fantastic, highly recommend.", 1),
    ("A delightful film with amazing performances.", 1),
    ("One of the best movies I have ever seen.", 1),
    ("The soundtrack and visuals are stunning.", 1),
    ("I was moved to tears in a good way.", 1),
    ("Clever, funny, and deeply satisfying.", 1),
    ("A masterpiece of modern cinema.", 1),
    ("A complete waste of time, absolutely terrible.", 0),
    ("Boring, predictable, and poorly acted.", 0),
    ("I hated this film from start to finish.", 0),
    ("The worst movie I have seen this year.", 0),
    ("Painfully slow and utterly pointless.", 0),
    ("Awful script and zero chemistry.", 0),
    ("I want my money back.", 0),
    ("Terrible direction and bad editing.", 0),
    ("Nothing made sense and it was dull.", 0),
    ("A disaster from beginning to end.", 0),
]

SAMPLES = [
    ("This movie is wonderful and heartwarming.", "positive"),
    ("A complete waste of time, absolutely terrible.", "negative"),
    ("The acting was okay but the plot was boring.", "negative"),
    ("I loved every minute of it!", "positive"),
]


def _quick_finetune(model, tokenizer, device):
    """3 epoch 迷你微调，不依赖 HuggingFace Hub。"""
    model.train()
    ids_list, mask_list, labels = [], [], []
    for text, lab in MINI_CORPUS:
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))
        labels.append(lab)
    input_ids = torch.stack(ids_list)
    attn = torch.stack(mask_list)
    y = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(input_ids, attn, y), batch_size=8, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        total, correct, loss_sum = 0, 0, 0.0
        for batch_ids, batch_mask, batch_y in loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_y)
            out.loss.backward()
            opt.step()
            loss_sum += out.loss.item()
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        print(f"  epoch {epoch+1}: loss={loss_sum/len(loader):.4f} acc={correct/total:.2f}")

    model.eval()


def _ensure_classifier(device):
    from transformers import BertForSequenceClassification, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    if os.path.isdir(CKPT_DIR) and os.path.isfile(os.path.join(CKPT_DIR, "config.json")):
        print(f"[模型] 从缓存加载: {CKPT_DIR}")
        model = BertForSequenceClassification.from_pretrained(CKPT_DIR)
        return model.to(device), tokenizer

    print("[模型] 首次运行：内置语料快速微调 BERT-Base（约 30 秒）...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    model.to(device)
    _quick_finetune(model, tokenizer, device)

    os.makedirs(CKPT_DIR, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    print(f"[模型] 已保存到 {CKPT_DIR}")
    return model, tokenizer


@torch.no_grad()
def bert_cls_embedding(model, tokenizer, text: str, device) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return model.bert(**inputs).pooler_output.cpu()  # (1, 768) CPU for CrypTen


def classify_plaintext(model, tokenizer, text: str, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu()
    pred = int(probs.argmax())
    labels = ["negative", "positive"]
    return labels[pred], probs.tolist()


def classify_crypten(model, tokenizer, text: str, device):
    import crypten

    patch_crypten_inprocess()
    crypten.init_thread(0, 1)

    cls = bert_cls_embedding(model, tokenizer, text, device)
    W = model.classifier.weight.detach().cpu().T
    b = model.classifier.bias.detach().cpu()

    cls_enc = crypten.cryptensor(cls)
    W_enc = crypten.cryptensor(W)
    b_enc = crypten.cryptensor(b)
    logits_enc = cls_enc.matmul(W_enc) + b_enc
    probs_enc = logits_enc.softmax(dim=-1)
    probs = probs_enc.get_plain_text()[0]
    pred = int(probs.argmax())
    labels = ["negative", "positive"]
    return labels[pred], probs.tolist()


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"错误: 未找到 {MODEL_DIR}")
        print("请先运行: python scripts/download_models.py")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    model, tokenizer = _ensure_classifier(device)

    print("\n" + "=" * 65)
    print("CrypTen + BERT 语义分类（情感二分类）")
    print("=" * 65)

    max_err = 0.0
    label_match = 0
    for text, expected in SAMPLES:
        plain_label, plain_probs = classify_plaintext(model, tokenizer, text, device)
        cipher_label, cipher_probs = classify_crypten(model, tokenizer, text, device)

        err = max(abs(a - b) for a, b in zip(plain_probs, cipher_probs))
        max_err = max(max_err, err)
        if plain_label == cipher_label:
            label_match += 1

        print(f"\n文本: {text}")
        print(f"  标注: {expected} | 明文: {plain_label} | 密文: {cipher_label}")
        print(f"  明文概率: {[f'{p:.4f}' for p in plain_probs]}")
        print(f"  密文概率: {[f'{p:.4f}' for p in cipher_probs]}")
        print(f"  概率差: {err:.2e}  {'[OK]' if plain_label == cipher_label else '[MISMATCH]'}")

    print("\n" + "-" * 65)
    print(f"明文/密文标签一致: {label_match}/{len(SAMPLES)}")
    print(f"概率最大差: {max_err:.2e}")
    ok = max_err < 1e-3 and label_match == len(SAMPLES)
    print("[OK] CrypTen+BERT 语义分类跑通" if ok else "[WARN] 请检查环境")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
