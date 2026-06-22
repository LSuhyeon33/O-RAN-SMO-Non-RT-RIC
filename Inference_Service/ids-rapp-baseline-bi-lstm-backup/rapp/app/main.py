import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import simplejson as json
from flask import Flask, request, jsonify

import config
import inference
import influx_client
import preprocess
import bench
from features import SCENARIO_TO_LABEL
from metrics import compute_metrics, save_metrics

app = Flask(__name__)

_RUN_LOCK = threading.Lock()
_RUN = {
    # idle | fetching | inferring | metrics | bench_latency | bench_resource
    #      | bench_retraining | completed | error
    "phase": "idle",
    "started_at": None,
    "completed_at": None,
    "batch": 0,
    "total_batches": 0,
    "elapsed_s": 0.0,
    "eta_s": None,
    "per_scenario_windows": {},
    "total_windows": 0,
    "error": None,
    "result_summary": None,
    "warmup_calls": 0,
    "batch_size": 0,
}

_BUSY_PHASES = ("fetching", "inferring", "metrics",
                "bench_latency", "bench_resource", "bench_retraining")


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _set(**kw):
    with _RUN_LOCK:
        for k, v in kw.items():
            _RUN[k] = v


def _progress_cb(batch_done, total_batches, started_at_perf):
    elapsed = time.perf_counter() - started_at_perf
    eta = (elapsed / batch_done) * (total_batches - batch_done) if batch_done > 0 else None
    with _RUN_LOCK:
        _RUN["batch"] = int(batch_done)
        _RUN["total_batches"] = int(total_batches)
        _RUN["elapsed_s"] = float(elapsed)
        _RUN["eta_s"] = float(eta) if eta is not None else None


def _run_evaluate(batch_size: int, warmup: int):
    t_perf0 = time.perf_counter()
    try:
        _set(phase="fetching", started_at=_iso_now(), completed_at=None,
             batch=0, total_batches=0, elapsed_s=0.0, eta_s=None,
             per_scenario_windows={}, total_windows=0, error=None,
             result_summary=None, batch_size=batch_size, warmup_calls=warmup)

        Xs, ys = [], []
        per_scen = {}
        for scen in SCENARIO_TO_LABEL.keys():
            df = influx_client.fetch_test_by_scenario(scen)
            X, y = preprocess.transform_windows_per_scenario(df, scenario=scen)
            per_scen[scen] = int(len(y))
            if len(y) > 0:
                Xs.append(X)
                ys.append(y)
            with _RUN_LOCK:
                _RUN["per_scenario_windows"] = dict(per_scen)
                _RUN["elapsed_s"] = float(time.perf_counter() - t_perf0)

        if not Xs:
            raise RuntimeError("no windows fetched from InfluxDB")

        X_all = np.concatenate(Xs, axis=0)
        y_true = np.concatenate(ys, axis=0)
        total_windows = int(len(y_true))
        total_batches = (total_windows + batch_size - 1) // batch_size
        _set(phase="inferring", total_windows=total_windows,
             total_batches=total_batches)

        t_inf0 = time.perf_counter()
        probs, latency_ms = inference.predict_batch(
            X_all, batch_size=batch_size, warmup=warmup,
            progress_cb=lambda d, t: _progress_cb(d, t, t_inf0),
        )

        _set(phase="metrics")
        m = compute_metrics(y_true, probs, latency_ms, batch_size)
        m["per_scenario_windows"] = per_scen
        m["warmup_calls"] = warmup
        save_metrics(m, latency_ms, config.RESULTS_DIR)

        summary = {
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "fnr_avg": m["fnr_avg"],
            "roc_auc_ovr": m["roc_auc_ovr"],
            "latency_ms_mean": m["latency_ms"]["mean"],
            "latency_ms_p95": m["latency_ms"]["p95"],
            "n_samples": m["n_samples"],
            "results_dir": config.RESULTS_DIR,
        }
        _set(phase="completed", completed_at=_iso_now(),
             elapsed_s=float(time.perf_counter() - t_perf0),
             eta_s=0.0, result_summary=summary)
        print(f"[evaluate] done: {json.dumps(summary, ignore_nan=True)}",
              file=sys.stderr, flush=True)
    except Exception as e:
        traceback.print_exc()
        _set(phase="error", error=str(e), completed_at=_iso_now(),
             elapsed_s=float(time.perf_counter() - t_perf0))


def _run_bench(phase: str, fn, kwargs: dict):
    """벤치 함수(fn)를 백그라운드에서 실행하고 _RUN 상태를 갱신한다."""
    t_perf0 = time.perf_counter()
    try:
        _set(phase=phase, started_at=_iso_now(), completed_at=None,
             batch=0, total_batches=0, elapsed_s=0.0, eta_s=None,
             error=None, result_summary=None)

        def status_cb(**kw):
            with _RUN_LOCK:
                for k, v in kw.items():
                    _RUN[k] = v
                _RUN["elapsed_s"] = float(time.perf_counter() - t_perf0)

        result = fn(status_cb=status_cb, **kwargs)
        _set(phase="completed", completed_at=_iso_now(),
             elapsed_s=float(time.perf_counter() - t_perf0), eta_s=0.0,
             result_summary=result)
        print(f"[{phase}] done -> {result.get('results_path')}",
              file=sys.stderr, flush=True)
    except Exception as e:
        traceback.print_exc()
        _set(phase="error", error=str(e), completed_at=_iso_now(),
             elapsed_s=float(time.perf_counter() - t_perf0))


def _start_bg(target, args):
    with _RUN_LOCK:
        if _RUN["phase"] in _BUSY_PHASES:
            return jsonify({"error": "a run is already in progress",
                            "status": dict(_RUN)}), 409
    threading.Thread(target=target, args=args, daemon=True).start()
    return None


@app.get('/healthz')
def healthz():
    scaler_loaded = False
    try:
        preprocess.get_scaler()
        scaler_loaded = True
    except Exception as e:
        print(f"[healthz] scaler error: {e}", file=sys.stderr)
    return jsonify({
        'model_loaded': inference.model_loaded(),
        'scaler_loaded': scaler_loaded,
        'influx_reachable': influx_client.health(),
        'window_size': config.WINDOW_SIZE,
        'model_name': config.MODEL_NAME,
    })


@app.get('/status')
def status():
    with _RUN_LOCK:
        return jsonify(dict(_RUN))


@app.post('/predict')
def predict_one():
    try:
        body = request.get_json(force=True)
        window = body.get('window')
        if window is None:
            return jsonify({'error': 'body must contain "window"'}), 400
        X = preprocess.transform_single_window(window)
        probs, latency_ms = inference.predict_single(X)
        return jsonify({
            'predicted_label': int(np.argmax(probs)),
            'probs': probs.tolist(),
            'latency_ms': latency_ms,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.post('/evaluate')
def evaluate():
    """Start an async evaluation. Returns immediately with status; poll /status."""
    with _RUN_LOCK:
        if _RUN["phase"] in _BUSY_PHASES:
            return jsonify({"error": "a run is already in progress", "status": dict(_RUN)}), 409
    body = request.get_json(silent=True) or {}
    batch_size = int(body.get('batch_size', 256))
    warmup = int(body.get('warmup_calls', 100))
    threading.Thread(target=_run_evaluate, args=(batch_size, warmup), daemon=True).start()
    return jsonify({
        "status": "started",
        "batch_size": batch_size,
        "warmup_calls": warmup,
        "poll": "/status",
    }), 202


@app.post('/bench/latency')
def bench_latency_ep():
    """① 추론 지연 — body: {batch_sizes, n_warmup, n_measure, mode}."""
    body = request.get_json(silent=True) or {}
    kwargs = {
        'batch_sizes': body.get('batch_sizes', [1, 10, 100, 200, 300]),
        'n_warmup': int(body.get('n_warmup', 100)),
        'n_measure': int(body.get('n_measure', 100)),
        'mode': body.get('mode', 'both'),
    }
    busy = _start_bg(_run_bench, ("bench_latency", bench.bench_latency, kwargs))
    if busy:
        return busy
    return jsonify({"status": "started", "bench": "latency",
                    "params": kwargs, "poll": "/status"}), 202


@app.post('/bench/resource')
def bench_resource_ep():
    """② 시스템 자원 사용량 — 1회 추론 중 CPU%·RSS(MB) (파라미터 없음)."""
    busy = _start_bg(_run_bench, ("bench_resource", bench.bench_resource, {}))
    if busy:
        return busy
    return jsonify({"status": "started", "bench": "resource",
                    "poll": "/status"}), 202


@app.post('/bench/retraining')
def bench_retraining_ep():
    """③ 재학습 중 운영 가용성 — CPU%·RSS 시계열 (1초 간격, 파라미터 없음)."""
    busy = _start_bg(
        _run_bench,
        ("bench_retraining", bench.bench_retraining_availability, {}))
    if busy:
        return busy
    return jsonify({"status": "started", "bench": "retraining",
                    "poll": "/status"}), 202


if __name__ == '__main__':
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    try:
        preprocess.get_scaler()
        print("[startup] scaler loaded", file=sys.stderr)
    except Exception as e:
        print(f"[startup] scaler load failed: {e}", file=sys.stderr)
    try:
        inference.get_model()
        print("[startup] model loaded", file=sys.stderr)
    except Exception as e:
        print(f"[startup] model load failed: {e}", file=sys.stderr)
    # threaded=True allows /status to be served while a worker thread runs
    app.run(host='0.0.0.0', port=config.HTTP_PORT, threaded=True)
