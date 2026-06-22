import io
import os
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests

import config
from features import FEATURES

_SCALER = None


def _load_scaler_from_bundle():
    """KServe가 서빙하는 모델 번들(Model.zip)에서 scaler.pkl을 받아 로드한다.
    모델과 같은 번들에서 꺼내므로 scaler·모델 버전이 항상 일치한다.
    한 번 받으면 SCALER_CACHE_PATH 에 캐시해 재호출 시 재다운로드를 피한다.
    """
    if os.path.exists(config.SCALER_CACHE_PATH):
        print(f"[scaler] cache hit: {config.SCALER_CACHE_PATH}", file=sys.stderr)
        return joblib.load(config.SCALER_CACHE_PATH)

    print(f"[scaler] downloading model bundle: {config.MODEL_ZIP_URL}", file=sys.stderr)
    resp = requests.get(config.MODEL_ZIP_URL, timeout=config.REQUEST_TIMEOUT_S)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # 번들 내 경로 구조와 무관하게 파일명이 scaler.pkl 인 항목을 찾는다
        matches = [n for n in zf.namelist()
                   if n.rsplit('/', 1)[-1] == config.SCALER_FILENAME]
        if not matches:
            raise FileNotFoundError(
                f"'{config.SCALER_FILENAME}' not in model bundle; entries={zf.namelist()}")
        os.makedirs(os.path.dirname(config.SCALER_CACHE_PATH) or '.', exist_ok=True)
        with zf.open(matches[0]) as src, open(config.SCALER_CACHE_PATH, 'wb') as dst:
            dst.write(src.read())
    print(f"[scaler] extracted '{matches[0]}' -> {config.SCALER_CACHE_PATH}", file=sys.stderr)
    return joblib.load(config.SCALER_CACHE_PATH)


def get_scaler():
    global _SCALER
    if _SCALER is None:
        # SCALER_PATH(로컬 override)가 지정돼 있으면 그 파일, 아니면 모델 번들에서.
        if config.SCALER_PATH and os.path.exists(config.SCALER_PATH):
            print(f"[scaler] local override: {config.SCALER_PATH}", file=sys.stderr)
            _SCALER = joblib.load(config.SCALER_PATH)
        else:
            _SCALER = _load_scaler_from_bundle()
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


def transform_rows_per_scenario(df: pd.DataFrame, scenario: str = ""):
    """트리 모델(RF, XGBoost)은 윈도윙 없이 한 행이 곧 한 샘플이다.
    scaler 적용 후 (N, 20) 행렬과 (N,) 라벨을 그대로 반환한다.
    """
    if len(df) == 0:
        return (np.empty((0, len(FEATURES)), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    df = df.sort_values('ts_ns').reset_index(drop=True)
    raw = df[FEATURES].astype(float).values
    raw = _clean_features(raw, scenario)
    X = get_scaler().transform(raw).astype(np.float32)   # (N, 20)
    y = df['label'].values.astype(np.int64)
    return X, y


def transform_single_row(row_values):
    """피처 20개짜리 한 행을 받아 scaler 적용 후 1-D (20,) 로 반환.
    inference.predict_single 이 [np.newaxis, ...] 로 (1, 20) 을 만든다.
    """
    arr = np.asarray(row_values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    expected = (1, len(FEATURES))
    if arr.shape != expected:
        raise ValueError(
            f"row shape {arr.shape} != expected ({len(FEATURES)},) or {expected}")
    arr = _clean_features(arr, "single")
    return get_scaler().transform(arr).astype(np.float32)[0]
