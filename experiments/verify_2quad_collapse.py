"""
verify_2quad_collapse.py
核心实验：证明 SecFormer 的 2Quad 近似导致 BERT-Large CoLA 精度归零
流程：微调CoLA -> 原始Softmax评测 -> 2Quad替换后评测 -> 对比
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
import types

MODEL_LARGE = './bert-large-uncased'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def preprocess_cola(tokenizer):
    ds = load_dataset('glue', 'cola')
    def prep(ex):
        return tokenizer(ex['sentence'], max_length=128,
                         truncation=True, padding='max_length')
    ds = ds.map(prep, batched=True)
    ds = ds.rename_column('label', 'labels')
    ds.set_format('torch',
        columns=['input_ids','attention_mask','token_type_ids','labels'])
    return ds


def metric_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'matthews': round(matthews_corrcoef(labels, preds)*100, 2)}


def two_quad_softmax(scores):
    """SecFormer的2Quad：用二次函数替换Softmax"""
    c = 1.0
    num = (scores + c) ** 2
    return num / num.sum(dim=-1, keepdim=True).clamp(min=1e-9)


def patch_with_2quad(model):
    """把模型所有attention层的softmax替换为2Quad"""
    for layer in model.bert.encoder.layer:
        self_attn = layer.attention.self

        def new_forward(self, hidden_states, attention_mask=None,
                        head_mask=None, encoder_hidden_states=None,
                        encoder_attention_mask=None, past_key_value=None,
                        output_attentions=False, **kwargs):
            q = self.transpose_for_scores(self.query(hidden_states))
            k = self.transpose_for_scores(self.key(hidden_states))
            v = self.transpose_for_scores(self.value(hidden_states))
            scores = torch.matmul(q, k.transpose(-1,-2))
            scores = scores / (self.attention_head_size ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            # ★ 2Quad 替换 Softmax ★
            probs = two_quad_softmax(scores)
            ctx = torch.matmul(probs, v)
            ctx = ctx.permute(0,2,1,3).contiguous()
            ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
            return (ctx,)

        self_attn.forward = types.MethodType(new_forward, self_attn)
    return model


def evaluate_model(model, ds, tag):
    args = TrainingArguments(
        output_dir='./tmp_eval', per_device_eval_batch_size=64,
        report_to='none', dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, compute_metrics=metric_fn)
    res = trainer.evaluate(ds['validation'])
    score = res['eval_matthews']
    print(f'  [{tag}] Matthews = {score}')
    return score


def main():
    print('='*55)
    print('核心实验：2Quad 近似导致 CoLA 精度归零验证')
    print('='*55)

    tokenizer = BertTokenizer.from_pretrained(MODEL_LARGE)
    ds = preprocess_cola(tokenizer)

    # ── 第1步：微调 BERT-Large CoLA ──
    print('\n[1/3] 微调 BERT-Large CoLA（约20-30分钟）...')
    model = BertForSequenceClassification.from_pretrained(
        MODEL_LARGE, num_labels=2).to(DEVICE)

    args = TrainingArguments(
        output_dir='./results/cola_finetuned',
        num_train_epochs=3, per_device_train_batch_size=16,
        per_device_eval_batch_size=64, learning_rate=1e-5,
        save_strategy='no', report_to='none',
        dataloader_num_workers=0, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args,
        train_dataset=ds['train'], eval_dataset=ds['validation'],
        compute_metrics=metric_fn)
    trainer.train()

    # ── 第2步：原始Softmax评测 ──
    print('\n[2/3] 原始 Softmax 评测...')
    score_original = evaluate_model(model, ds, '原始Softmax')

    # ── 第3步：2Quad替换后评测 ──
    print('\n[3/3] 替换为 2Quad 后评测...')
    model_2quad = patch_with_2quad(model)
    score_2quad = evaluate_model(model_2quad, ds, '2Quad近似')

    # ── 结论 ──
    print('\n' + '='*55)
    print('实验结论')
    print('='*55)
    print(f'  BERT-Large CoLA 原始Softmax:  {score_original}')
    print(f'  BERT-Large CoLA 2Quad近似:    {score_2quad}')
    print(f'  精度损失:                      {round(score_original - score_2quad, 2)}')
    if score_2quad < 5:
        print('\n  ✓ 验证成功：2Quad近似导致CoLA精度归零')
        print('  ✓ 这证明了MCU精确Softmax的必要性')

    # 保存结果
    with open('results/2quad_collapse.txt', 'w', encoding='utf-8') as f:
        f.write(f'BERT-Large CoLA original_softmax: {score_original}\n')
        f.write(f'BERT-Large CoLA 2quad: {score_2quad}\n')
    print('\n结果已保存到 results/2quad_collapse.txt')


if __name__ == '__main__':
    main()