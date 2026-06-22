import csv
import json
import os
from typing import List

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

from features import N_CLASSES


def _percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float('nan')


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, latency_ms: List[float],
                    batch_size: int):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    fnr_per_class = []
    for i in range(N_CLASSES):
        fn = int(cm[i].sum() - cm[i, i])
        tp = int(cm[i, i])
        denom = fn + tp
        fnr_per_class.append(float(fn / denom) if denom > 0 else 0.0)
    fnr_avg = float(np.mean(fnr_per_class))

    auc_ovr = None
    try:
        present = set(int(x) for x in y_true)
        if len(present) == N_CLASSES:
            auc_ovr = float(roc_auc_score(
                y_true, y_prob, multi_class='ovr',
                labels=list(range(N_CLASSES))))
    except Exception as e:
        print(f"[warn] ROC-AUC failed: {e}")

    per_class = []
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0)
    for i in range(N_CLASSES):
        per_class.append({
            'label': i,
            'precision': float(p_arr[i]),
            'recall': float(r_arr[i]),
            'f1': float(f1_arr[i]),
            'fnr': fnr_per_class[i],
            'support': int(sup_arr[i]),
        })

    lat_arr = np.asarray(latency_ms, dtype=float)
    latency = {
        'mean': float(lat_arr.mean()) if lat_arr.size else None,
        'std': float(lat_arr.std()) if lat_arr.size else None,
        'p50': _percentile(lat_arr, 50),
        'p95': _percentile(lat_arr, 95),
        'p99': _percentile(lat_arr, 99),
        'n_batches': int(lat_arr.size),
        'batch_size': int(batch_size),
    }

    metrics = {
        'accuracy': acc,
        'precision_macro': float(p_m), 'precision_weighted': float(p_w),
        'recall_macro': float(r_m), 'recall_weighted': float(r_w),
        'f1_macro': float(f1_m), 'f1_weighted': float(f1_w),
        'fnr_avg': fnr_avg,
        'fnr_per_class': fnr_per_class,
        'roc_auc_ovr': auc_ovr,
        'latency_ms': latency,
        'confusion_matrix': cm.tolist(),
        'per_class': per_class,
        'n_samples': int(len(y_true)),
        's2_leakage_note': "label 3 (S2) test rows fully overlap with training; FNR/accuracy reference only",
    }
    return metrics


def save_metrics(metrics: dict, latency_ms: List[float], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, 'confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['actual\\predicted'] + [f'pred_{i}' for i in range(N_CLASSES)])
        for i, row in enumerate(metrics['confusion_matrix']):
            w.writerow([f'actual_{i}'] + row)

    with open(os.path.join(out_dir, 'per_class.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'precision', 'recall', 'f1', 'fnr', 'support'])
        for row in metrics['per_class']:
            w.writerow([row['label'], row['precision'], row['recall'],
                        row['f1'], row['fnr'], row['support']])

    with open(os.path.join(out_dir, 'latency_samples.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch_idx', 'latency_ms'])
        for i, v in enumerate(latency_ms):
            w.writerow([i, v])
