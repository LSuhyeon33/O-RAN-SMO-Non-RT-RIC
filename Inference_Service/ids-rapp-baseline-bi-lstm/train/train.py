#!/usr/bin/env python3
"""Baseline Bi-LSTM 로컬 학습 CLI.

icsr-pipelines/oom_pipeline_bilstm.py 의 최종 모델 학습 로직을 KFP/FeatureStore
의존성 없이 단일 로컬 실행으로 포팅한 것.  하이퍼파라미터는 이미 Exp 1에서
결정되었으므로 CV는 생략하고 validation_split 으로 EarlyStopping 만 적용한다.

산출물: <output-dir>/model.keras + <output-dir>/scaler.pkl
이 두 파일은 Dockerfile 이 /model 로 베이크하여 rApp 이 in-process 로 로드한다.
"""
import argparse
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd

# FEATURES 는 rApp 과 동일 출처(rapp/app/features.py)에서 가져온다.
# 컨테이너(/app/train.py + /app/features.py)와 레포(train/ + rapp/app/) 양쪽 지원.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, '..', 'rapp', 'app')):
    if os.path.isfile(os.path.join(_cand, 'features.py')):
        sys.path.insert(0, _cand)
        break
from features import FEATURES, LABEL, N_CLASSES, N_FEATURES  # noqa: E402

TS_COL = 'ts_ns'
DEFAULT_CSV = ('/usr/local/o-ran/Dataset/OOM dataset/processed/'
               'balanced_train_classification_dataset.csv')


def make_windows(X, y, w):
    """(N, feat) → (N-w+1, w, feat) 슬라이딩 윈도우, 라벨은 윈도우 마지막 시점."""
    n = len(X) - w + 1
    Xs = np.empty((n, w, X.shape[1]), dtype=np.float32)
    ys = np.empty((n,), dtype=np.int64)
    for i in range(n):
        Xs[i] = X[i:i + w]
        ys[i] = y[i + w - 1]
    return Xs, ys


def build_model(window, hidden, layers, dropout):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    m = Sequential()
    for li in range(int(layers)):
        return_seq = (li < int(layers) - 1)
        if li == 0:
            m.add(Bidirectional(
                LSTM(int(hidden), activation='tanh', return_sequences=return_seq),
                input_shape=(int(window), N_FEATURES)))
        else:
            m.add(Bidirectional(
                LSTM(int(hidden), activation='tanh', return_sequences=return_seq)))
        m.add(Dropout(float(dropout)))
    m.add(Dense(N_CLASSES, activation='softmax'))
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              jit_compile=False)
    m.summary()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', default=DEFAULT_CSV)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default='model')
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    print(f"[load] {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    print(f"[load] rows={len(df)}")

    df[TS_COL] = pd.to_numeric(df[TS_COL], errors='coerce')
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)
    print(f"[sort] sorted by {TS_COL}, rows={len(df)}")

    X_df = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y_arr = df[LABEL].astype(int).values
    X_arr = X_df.values

    # scaler: 2D 피처 전체에 fit (Proposed preprocess.transform 과 동일 좌표계)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    X_seq, y_seq = make_windows(X_scaled, y_arr, int(args.window))
    print(f"[window] X_seq={X_seq.shape}, y_seq={y_seq.shape}")

    present = np.unique(y_seq)
    cw = compute_class_weight('balanced', classes=present, y=y_seq)
    class_weight = {int(c): float(w) for c, w in zip(present, cw)}
    for c in range(N_CLASSES):
        class_weight.setdefault(c, 1.0)
    print(f"[class_weight] {class_weight}")

    model = build_model(args.window, args.hidden, args.layers, args.dropout)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=int(args.epochs), batch_size=int(args.batch),
        class_weight=class_weight,
        callbacks=[es],
        verbose=2,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.keras')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[save] {model_path}")
    print(f"[save] {scaler_path}")


if __name__ == '__main__':
    main()
