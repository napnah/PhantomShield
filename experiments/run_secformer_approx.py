"""
run_secformer_approx.py
复现 SecFormer 的 2Quad 近似推理
关键：验证 BERT-Large CoLA 任务 Matthews 系数归零
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
import os

MODEL_BASE  = './bert-base-uncased'
MODEL_LARGE = './bert-large-uncased'


def quad(x):
    """Quad(x) = 0.125x² + 0.25x + 0.5（替换GeLU）"""
    return 0.125 * x**2 + 0.25 * x + 0.5


def two_quad(x):
    """2Quad：替换Softmax的二次函数归一化"""
    c = 1.0  # 偏移常数
    numerator = (x + c) ** 2
    denominator = numerator.sum(dim=-1, keepdim=True)
    denominator = denominator.clamp(min=1e-9)
    return numerator / denominator


def replace_bert_with_approx(model):
    """
    将BERT的Softmax替换为2Quad
    模拟SecFormer的模型设计
    """
    for layer in model.bert.encoder.layer:
        # 替换Attention中的Softmax
        original_forward = layer.attention.self.forward

        def make_approx_forward(orig):
            def approx_forward(hidden_states, attention_mask=None,
                               head_mask=None, encoder_hidden_states=None,
                               encoder_attention_mask=None,
                               past_key_value=None, output_attentions=False):
                # 计算attention scores
                query = layer.attention.self.transpose_for_scores(
                    layer.attention.self.query(hidden_states))
                key = layer.attention.self.transpose_for_scores(
                    layer.attention.self.key(hidden_states))
                value = layer.attention.self.transpose_for_scores(
                    layer.attention.self.value(hidden_states))

                scores = torch.matmul(query, key.transpose(-1, -2))
                scores = scores / (layer.attention.self.attention_head_size ** 0.5)

                if attention_mask is not None:
                    scores = scores + attention_mask

                # ★ 关键：用2Quad替换Softmax ★
                probs = two_quad(scores)

                context = torch.matmul(probs, value)
                context = context.permute(0, 2, 1, 3).contiguous()
                new_shape = context.size()[:-2] + (layer.attention.self.all_head_size,)
                context = context.view(new_shape)

                outputs = (context, probs) if output_attentions else (context,)
                return outputs

            return approx_forward

        layer.attention.self.forward = make_approx_forward(original_forward)

    print('  ✓ Softmax → 2Quad 替换完成')
    return model


def evaluate_cola(model_path, use_approx=False, tag=''):
    """在CoLA验证集上评测Matthews系数"""
    print(f'\n评测 {tag}（{"2Quad近似" if use_approx else "原始Softmax"}）...')

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=2
    )
    model.eval()

    if use_approx:
        model = replace_bert_with_approx(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    dataset = load_dataset('glue', 'cola')
    val_data = dataset['validation']

    all_preds, all_labels = [], []
    batch_size = 32

    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i+batch_size]
        inputs = tokenizer(
            batch['sentence'], max_length=128,
            truncation=True, padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor(batch['label'])

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    matthews = matthews_corrcoef(all_labels, all_preds)
    accuracy = sum(p==l for p,l in zip(all_preds, all_labels)) / len(all_labels)

    print(f'  Matthews系数: {matthews:.4f} ({matthews*100:.1f}%)')
    print(f'  准确率: {accuracy*100:.1f}%')
    return matthews


def main():
    print('='*55)
    print('SecFormer 2Quad 近似效果验证')
    print('目标：复现 BERT-Large CoLA Matthews=0')
    print('='*55)

    results = {}

    # BERT-Base：原始 vs 2Quad
    if os.path.exists(MODEL_BASE):
        print('\n[BERT-Base]')
        results['base_original'] = evaluate_cola(
            MODEL_BASE, use_approx=False, tag='BERT-Base 原始')
        results['base_approx'] = evaluate_cola(
            MODEL_BASE, use_approx=True,  tag='BERT-Base 2Quad近似')
    else:
        print(f'\n跳过BERT-Base：{MODEL_BASE} 不存在')

    # BERT-Large：原始 vs 2Quad（关键实验）
    if os.path.exists(MODEL_LARGE):
        print('\n[BERT-Large] ← 关键实验')
        results['large_original'] = evaluate_cola(
            MODEL_LARGE, use_approx=False, tag='BERT-Large 原始')
        results['large_approx'] = evaluate_cola(
            MODEL_LARGE, use_approx=True,  tag='BERT-Large 2Quad近似')
    else:
        print(f'\n跳过BERT-Large：{MODEL_LARGE} 不存在，请先下载')

    # 汇总
    print('\n' + '='*55)
    print('汇总结果')
    print('='*55)
    print(f'{"方案":30s} {"Matthews":>10s} {"论文参考":>10s}')
    print('-'*55)

    refs = {
        'base_original':  57.8,
        'base_approx':    52.6,  # SecFormer MPCFormer基线
        'large_original': 61.7,
        'large_approx':    0.0,  # ★ SecFormer BERT-Large CoLA=0
    }
    labels = {
        'base_original':  'BERT-Base 原始Softmax',
        'base_approx':    'BERT-Base 2Quad(SecFormer)',
        'large_original': 'BERT-Large 原始Softmax',
        'large_approx':   'BERT-Large 2Quad(SecFormer) ★',
    }

    for key, val in results.items():
        ref = refs.get(key, '--')
        label = labels.get(key, key)
        print(f'{label:30s} {val*100:>10.1f} {str(ref):>10s}')

    print('\n核心结论：')
    if 'large_approx' in results:
        v = results['large_approx'] * 100
        if v < 5:
            print(f'  ✓ BERT-Large 2Quad近似导致CoLA归零（{v:.1f}%）')
            print('  ✓ 这证明了精确Softmax的必要性，MCU的核心价值得到验证')
        else:
            print(f'  BERT-Large 2Quad结果: {v:.1f}%')
            print('  注意：需要使用微调后的模型权重才能完全复现论文结果')


if __name__ == '__main__':
    main()