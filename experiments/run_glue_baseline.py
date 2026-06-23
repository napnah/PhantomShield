"""
run_glue_baseline.py
Reproduce SecFormer Table 2 GLUE baseline
Key goal: BERT-Large CoLA Matthews score
"""
import torch
import numpy as np
import csv
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

MODEL_BASE  = './bert-base-uncased'
MODEL_LARGE = './bert-large-uncased'

TASK_CONFIG = {
    'rte':  {'keys': ('sentence1','sentence2'), 'metric': 'accuracy', 'num_labels': 2},
    'mrpc': {'keys': ('sentence1','sentence2'), 'metric': 'f1',       'num_labels': 2},
    'cola': {'keys': ('sentence',),             'metric': 'matthews',  'num_labels': 2},
    'stsb': {'keys': ('sentence1','sentence2'), 'metric': 'pearson',   'num_labels': 1},
    'qnli': {'keys': ('question','sentence'),   'metric': 'accuracy',  'num_labels': 2},
}

SECFORMER_REF = {
    'bert-base':  {'qnli': 91.2, 'cola': 57.1, 'stsb': 87.4, 'mrpc': 89.2, 'rte': 69.0},
    'bert-large': {'qnli': 92.0, 'cola': 61.3, 'stsb': 89.2, 'mrpc': 88.7, 'rte': 72.6}
}


def compute_metrics(task):
    def _compute(eval_pred):
        logits, labels = eval_pred
        if task == 'stsb':
            preds = logits.squeeze()
            p, _ = pearsonr(preds, labels)
            s, _ = spearmanr(preds, labels)
            return {'combined': round((float(p)+float(s))/2*100, 1)}
        preds = np.argmax(logits, axis=1)
        if task == 'cola':
            return {'matthews': round(matthews_corrcoef(labels, preds)*100, 1)}
        if task == 'mrpc':
            return {'f1': round(f1_score(labels, preds)*100, 1)}
        return {'accuracy': round((preds == labels).mean()*100, 1)}
    return _compute


def run_task(task_name, model_path, model_tag):
    print('\n' + '='*55)
    print('Task: ' + task_name.upper() + '  Model: ' + model_tag)
    print('='*55)

    cfg = TASK_CONFIG[task_name]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=cfg['num_labels'])

    dataset = load_dataset('glue', task_name)

    def preprocess(examples):
        keys = cfg['keys']
        texts = (examples[keys[0]],) if len(keys)==1 \
                else (examples[keys[0]], examples[keys[1]])
        return tokenizer(*texts, max_length=128,
                         truncation=True, padding='max_length')

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format('torch',
        columns=['input_ids','attention_mask','token_type_ids','labels'])

    args = TrainingArguments(
        output_dir='./results/' + model_tag + '/' + task_name,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        eval_strategy='epoch',
        save_strategy='no',
        report_to='none',
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics(task_name),
    )

    trainer.train()
    results = trainer.evaluate()

    key = {'accuracy':'eval_accuracy', 'f1':'eval_f1',
           'matthews':'eval_matthews', 'pearson':'eval_combined'}[cfg['metric']]
    score = results.get(key, 0)

    ref_key = 'bert-base' if 'base' in model_tag else 'bert-large'
    ref = SECFORMER_REF[ref_key].get(task_name, 0)
    print('\nResult: ' + str(score) + '  Reference: ' + str(ref))
    return score


def main():
    os.makedirs('results', exist_ok=True)
    all_results = {}

    experiments = [
        ('cola', MODEL_LARGE, 'bert-large'),
        ('cola', MODEL_BASE,  'bert-base'),
        ('rte',  MODEL_BASE,  'bert-base'),
        ('mrpc', MODEL_BASE,  'bert-base'),
        ('stsb', MODEL_BASE,  'bert-base'),
        ('qnli', MODEL_BASE,  'bert-base'),
    ]

    csv_rows = []
    for task, model_path, model_tag in experiments:
        if not os.path.exists(model_path):
            print('\nSkip ' + task + '/' + model_tag + ': model not found')
            continue
        key = model_tag + '/' + task
        try:
            score = run_task(task, model_path, model_tag)
            all_results[key] = score
            csv_rows.append({'model': model_tag, 'task': task, 'score': score})
        except Exception as e:
            print('Failed: ' + str(e))
            all_results[key] = 'ERROR'

        with open('results/baseline_results.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['model','task','score'])
            writer.writeheader()
            writer.writerows(csv_rows)

    print('\n\n' + '='*55)
    print('GLUE Baseline Summary')
    print('='*55)
    for key, score in all_results.items():
        model_tag, task = key.split('/')
        ref_key = 'bert-base' if 'base' in model_tag else 'bert-large'
        ref = SECFORMER_REF[ref_key].get(task, '--')
        print(key.ljust(20) + ' score=' + str(score) + ' ref=' + str(ref))
    print('\nSaved to results/baseline_results.csv')


if __name__ == '__main__':
    main()