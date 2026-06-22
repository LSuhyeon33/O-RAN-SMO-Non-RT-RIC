import sys
import time

import numpy as np

import config

# Baseline: 모델을 rApp 프로세스 안에서 직접 실행한다 (KServe HTTP 호출 없음).
# tensorflow 는 무겁고 GPU 초기화를 동반하므로 최초 추론 시점에 lazy 로드한다.
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import tensorflow as tf  # lazy import — 모듈 import 시 TF/GPU 초기화 회피
        print(f"[model] loading {config.MODEL_PATH}", file=sys.stderr, flush=True)
        _MODEL = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        print("[model] loaded", file=sys.stderr, flush=True)
    return _MODEL


def reload_model():
    """디스크의 모델을 다시 로드한다 (재학습 후 모델 교체용)."""
    global _MODEL
    _MODEL = None
    return get_model()


def model_loaded() -> bool:
    """이미 메모리에 로드되었거나, 로드 가능한 산출물이 존재하면 True (/healthz용)."""
    if _MODEL is not None:
        return True
    import os
    return os.path.exists(config.MODEL_PATH)


def predict_batch(X: np.ndarray, batch_size: int = 256, warmup: int = 0,
                  progress_cb=None):
    """Run inference over X (N, window, features). Returns (probs, latency_ms_list).

    Proposed(inference.py)와 동일한 시그니처/리턴 → main.py·metrics.py 수정 불필요.
    latency_ms_list 는 배치 단위 호출 지연(ms) 리스트.
    progress_cb(batches_done, total_batches) 는 배치마다 호출.
    """
    if X.size == 0:
        return np.empty((0,), dtype=np.float32), []

    m = get_model()

    if warmup > 0:
        for _ in range(warmup):
            m.predict(X[:1], verbose=0)

    total_batches = (len(X) + batch_size - 1) // batch_size
    probs = []
    latencies = []
    for batch_idx, i in enumerate(range(0, len(X), batch_size)):
        chunk = X[i:i + batch_size]
        t0 = time.perf_counter()
        out = m.predict(chunk, verbose=0)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        probs.extend(np.asarray(out).tolist())
        latencies.append(dt_ms)
        if progress_cb is not None:
            progress_cb(batch_idx + 1, total_batches)

    return np.asarray(probs, dtype=np.float32), latencies


def predict_single(window: np.ndarray):
    """Run inference for one (window, features) sample. Returns (probs_vec, latency_ms)."""
    m = get_model()
    t0 = time.perf_counter()
    out = m.predict(window[np.newaxis, ...], verbose=0)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    probs = np.asarray(out[0], dtype=np.float32)
    return probs, dt_ms
