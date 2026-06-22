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
from features import SCENARIO_TO_LABEL
from metrics import compute_metrics, save_metrics
from system_monitor import SystemSampler

app = Flask(__name__)

_RUN_LOCK = threading.Lock()
_RUN = {
    "phase": "idle",          # idle | fetching | inferring | metrics | completed | error
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
    sampler = SystemSampler(interval_s=config.SYSTEM_SAMPLE_INTERVAL_S)
    sampler.start()
    try:
        _set(phase="fetching", started_at=_iso_now(), completed_at=None,
             batch=0, total_batches=0, elapsed_s=0.0, eta_s=None,
             per_scenario_windows={}, total_windows=0, error=None,
             result_summary=None, batch_size=batch_size, warmup_calls=warmup)
        sampler.set_phase("fetching")

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
        sampler.set_phase("inferring")

        t_inf0 = time.perf_counter()
        probs, latency_ms = inference.predict_batch(
            X_all, batch_size=batch_size, warmup=warmup,
            progress_cb=lambda d, t: _progress_cb(d, t, t_inf0),
        )

        _set(phase="metrics")
        sampler.set_phase("metrics")
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
    finally:
        # 성공·에러 모두에서 system_samples.csv 가 남도록 stop + save (best-effort)
        sampler.stop()
        try:
            sampler.save(config.RESULTS_DIR)
        except Exception as e:
            print(f"[evaluate] system_samples 저장 실패: {e}",
                  file=sys.stderr, flush=True)


@app.get('/healthz')
def healthz():
    scaler_loaded = False
    try:
        preprocess.get_scaler()
        scaler_loaded = True
    except Exception as e:
        print(f"[healthz] scaler error: {e}", file=sys.stderr)
    return jsonify({
        'scaler_loaded': scaler_loaded,
        'kserve_reachable': inference.kserve_reachable(),
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
    """Start an async evaluation. Returns immediately with status; poll /status for progress."""
    with _RUN_LOCK:
        if _RUN["phase"] in ("fetching", "inferring", "metrics"):
            return jsonify({"error": "evaluation already running", "status": dict(_RUN)}), 409
    body = request.get_json(silent=True) or {}
    batch_size = int(body.get('batch_size', 256))
    warmup = int(body.get('warmup_calls', 100))
    t = threading.Thread(target=_run_evaluate, args=(batch_size, warmup), daemon=True)
    t.start()
    return jsonify({
        "status": "started",
        "batch_size": batch_size,
        "warmup_calls": warmup,
        "poll": "/status",
    }), 202


if __name__ == '__main__':
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    try:
        preprocess.get_scaler()
        print("[startup] scaler loaded", file=sys.stderr)
    except Exception as e:
        print(f"[startup] scaler load failed: {e}", file=sys.stderr)
    # threaded=True allows /status to be served while /evaluate worker thread runs
    app.run(host='0.0.0.0', port=config.HTTP_PORT, threaded=True)
