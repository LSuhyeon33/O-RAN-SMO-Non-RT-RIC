import time
import numpy as np
import requests

import config


def kserve_reachable() -> bool:
    try:
        r = requests.get(config.MODEL_META_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def predict_batch(X: np.ndarray, batch_size: int = 256, warmup: int = 0,
                  progress_cb=None):
    """Run inference over X (N, features). Returns (probs, latency_ms_list).

    progress_cb(batches_done, total_batches) is called after each batch if provided.
    """
    if X.size == 0:
        return np.empty((0,), dtype=np.float32), []

    if warmup > 0:
        dummy_payload = {
            "signature_name": "serving_default",
            "instances": X[:1].tolist(),
        }
        for _ in range(warmup):
            requests.post(config.PREDICT_URL, json=dummy_payload, timeout=config.REQUEST_TIMEOUT_S)

    total_batches = (len(X) + batch_size - 1) // batch_size
    probs = []
    latencies = []
    for batch_idx, i in enumerate(range(0, len(X), batch_size)):
        chunk = X[i:i + batch_size]
        payload = {
            "signature_name": "serving_default",
            "instances": chunk.tolist(),
        }
        t0 = time.perf_counter()
        r = requests.post(config.PREDICT_URL, json=payload, timeout=config.REQUEST_TIMEOUT_S)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        probs.extend(r.json()['predictions'])
        latencies.append(dt_ms)
        if progress_cb is not None:
            progress_cb(batch_idx + 1, total_batches)

    return np.asarray(probs, dtype=np.float32), latencies


def predict_single(row: np.ndarray):
    """Run inference for one (features,) row. Returns (probs_vec, latency_ms)."""
    payload = {
        "signature_name": "serving_default",
        "instances": row[np.newaxis, ...].tolist(),
    }
    t0 = time.perf_counter()
    r = requests.post(config.PREDICT_URL, json=payload, timeout=config.REQUEST_TIMEOUT_S)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    probs = np.asarray(r.json()['predictions'][0], dtype=np.float32)
    return probs, dt_ms
