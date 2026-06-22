import sys

import joblib
import numpy as np
import pandas as pd

import config
from features import FEATURES

# Baseline: scaler 는 학습(train.py)이 만든 로컬 파일을 그대로 쓴다.
# Proposed 의 Model.zip 다운로드(_load_scaler_from_bundle)는 제거.
_SCALER = None


def get_scaler():
    global _SCALER
    if _SCALER is None:
        print(f"[scaler] loading {config.SCALER_PATH}", file=sys.stderr)
        _SCALER = joblib.load(config.SCALER_PATH)
    return _SCALER


def _clean_features(values: np.ndarray, scenario: str = "") -> np.ndarray:
    """Match the training pipeline's pd.to_numeric(errors='coerce').fillna(0.0).
    Replaces NaN/Inf in raw feature values with 0.0 before scaler.transform.
    """
    bad = ~np.isfinite(values)
    n_bad = int(bad.sum())
    if n_bad > 0:
        print(f"[preprocess] {scenario or '?'}: {n_bad} non-finite raw values -> 0.0",
              file=sys.stderr, flush=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def transform_windows_per_scenario(df: pd.DataFrame, window: int = None,
                                   scenario: str = ""):
    if window is None:
        window = config.WINDOW_SIZE
    if len(df) < window:
        return (np.empty((0, window, len(FEATURES)), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    df = df.sort_values('ts_ns').reset_index(drop=True)
    raw = df[FEATURES].astype(float).values
    raw = _clean_features(raw, scenario)
    X_raw = get_scaler().transform(raw).astype(np.float32)
    y_arr = df['label'].values.astype(np.int64)

    n = len(X_raw) - window + 1
    Xs = np.empty((n, window, len(FEATURES)), dtype=np.float32)
    ys = np.empty((n,), dtype=np.int64)
    for i in range(n):
        Xs[i] = X_raw[i:i + window]
        ys[i] = y_arr[i + window - 1]
    return Xs, ys


def transform_single_window(window_values):
    arr = np.asarray(window_values, dtype=np.float64)
    expected = (config.WINDOW_SIZE, len(FEATURES))
    if arr.shape != expected:
        raise ValueError(f"window shape {arr.shape} != expected {expected}")
    arr = _clean_features(arr, "single")
    return get_scaler().transform(arr).astype(np.float32)
