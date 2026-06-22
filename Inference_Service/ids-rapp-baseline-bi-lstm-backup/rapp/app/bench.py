"""Table 10 (v3) 벤치마크 헬퍼 — Baseline rApp (in-process 모델).

①  bench_latency               추론 지연 (방식 A 배치단위 / 방식 B 누적)
②  bench_resource              1회 추론 중 CPU%·RSS(MB)
③  bench_retraining_availability  재학습 트리거 전후 CPU%·RSS 시계열 (운영 가용성)

모든 결과는 RESULTS_DIR/bench_<name>_<ts>.json 으로 저장한다.
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

import config
import inference
from features import N_FEATURES


# --------------------------------------------------------------------------- #
# 공통 유틸
# --------------------------------------------------------------------------- #
def _ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _stats(samples):
    """latency 샘플 리스트 → mean/std/p50/p95/p99/n (ms)."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return {'mean': None, 'std': None, 'p50': None, 'p95': None,
                'p99': None, 'n': 0}
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'n': int(arr.size),
    }


def _save(name: str, payload: dict) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f'bench_{name}_{_ts()}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] saved {path}", file=sys.stderr, flush=True)
    return path


def _dummy_windows(n: int):
    """(n, WINDOW_SIZE, N_FEATURES) 더미 입력 — 지연 측정 전용(라벨 무관)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(
        (n, config.WINDOW_SIZE, N_FEATURES)).astype(np.float32)


class _ResourceSampler:
    """백그라운드에서 현재 프로세스 CPU%·RSS 를 일정 간격으로 샘플링."""

    def __init__(self, interval_s: float = 1.0):
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.interval = interval_s
        self.samples = []          # [{t, cpu_percent, rss_mb}]
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._phase = 'idle'
        self.proc.cpu_percent(None)   # 첫 호출은 0.0 → 기준선 prime

    def set_phase(self, phase: str):
        self._phase = phase

    def _loop(self):
        while not self._stop.is_set():
            cpu = self.proc.cpu_percent(None)
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            self.samples.append({
                't': round(time.perf_counter() - self._t0, 3),
                'phase': self._phase,
                'cpu_percent': float(cpu),
                'rss_mb': float(rss_mb),
            })
            self._stop.wait(self.interval)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)


# --------------------------------------------------------------------------- #
# ① 추론 지연
# --------------------------------------------------------------------------- #
def bench_latency(batch_sizes=None, n_warmup=100, n_measure=100, mode='both',
                  status_cb=None):
    """배치 크기별 추론 지연을 두 방식으로 측정한다.

    [방식 A] 배치 단위 호출 — batch_size 만큼 묶어 1회 predict, 응답 시간 기록.
             분포(mean/p95/p99/std) 산출을 위해 n_measure 회 반복 측정한다.
    [방식 B] 누적 호출      — 1건씩 batch_size 번 반복 predict, 호출별 지연 +
             누적 시간 기록 (배치 이득 없는 단건 처리 비용).

    n_measure 는 방식 A 의 반복 횟수(분포 산출용, 기본 100). 방식 B 는 단건을
    batch_size 번 호출하므로 그 자체로 batch_size 개 표본을 만든다.
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 200, 300]
    inference.get_model()
    pool = _dummy_windows(max(batch_sizes))

    results = {}
    for bi, b in enumerate(batch_sizes):
        entry = {}
        batch = pool[:b]

        if mode in ('both', 'a'):
            for _ in range(n_warmup):
                inference.predict_batch(batch[:1], batch_size=1)
            a_samples = []
            for _ in range(n_measure):
                t0 = time.perf_counter()
                inference.predict_batch(batch, batch_size=b)
                a_samples.append((time.perf_counter() - t0) * 1000.0)
            entry['method_a_batched'] = _stats(a_samples)

        if mode in ('both', 'b'):
            for _ in range(n_warmup):
                inference.predict_single(pool[0])
            b_samples = []
            t_cum0 = time.perf_counter()
            for _ in range(b):
                t0 = time.perf_counter()
                inference.predict_single(pool[0])
                b_samples.append((time.perf_counter() - t0) * 1000.0)
            cum_ms = (time.perf_counter() - t_cum0) * 1000.0
            sb = _stats(b_samples)
            sb['cumulative_ms'] = float(cum_ms)
            entry['method_b_accumulated'] = sb

        results[str(b)] = entry
        if status_cb:
            status_cb(batch=bi + 1, total_batches=len(batch_sizes))

    payload = {
        'name': 'latency',
        'model_name': config.MODEL_NAME,
        'window_size': config.WINDOW_SIZE,
        'batch_sizes': batch_sizes,
        'n_warmup': n_warmup,
        'n_measure': n_measure,
        'mode': mode,
        'results': results,
    }
    path = _save('latency', payload)
    payload['results_path'] = path
    return payload


# --------------------------------------------------------------------------- #
# ② 시스템 자원 사용량 (1회 추론 중)
# --------------------------------------------------------------------------- #
def bench_resource(status_cb=None):
    """1회 추론 요청을 수행하는 동안 rApp 프로세스 CPU%·RSS(MB)를 샘플링."""
    inference.get_model()
    sampler = _ResourceSampler(interval_s=0.05)
    sampler.set_phase('inference')

    win = _dummy_windows(1)[0]
    inference.predict_single(win)             # 워밍업 1회 (그래프 초기화 비용 제외)

    sampler.start()
    _, latency_ms = inference.predict_single(win)
    time.sleep(0.1)                           # 샘플 몇 개 확보
    sampler.stop()

    cpu = [s['cpu_percent'] for s in sampler.samples]
    rss = [s['rss_mb'] for s in sampler.samples]
    payload = {
        'name': 'resource',
        'model_name': config.MODEL_NAME,
        'inference_latency_ms': float(latency_ms),
        'cpu_percent': {
            'mean': float(np.mean(cpu)) if cpu else None,
            'peak': float(np.max(cpu)) if cpu else None,
        },
        'rss_mb': {
            'mean': float(np.mean(rss)) if rss else None,
            'peak': float(np.max(rss)) if rss else None,
        },
        'samples': sampler.samples,
    }
    path = _save('resource', payload)
    payload['results_path'] = path
    return payload


# --------------------------------------------------------------------------- #
# ③ 재학습 중 운영 가용성
# --------------------------------------------------------------------------- #
def bench_retraining_availability(normal_s=10, resume_s=10, status_cb=None):
    """재학습 트리거 전후 rApp CPU%·RSS 를 1초 간격으로 연속 모니터링.

    구간: 정상 추론(normal_s) → 재학습 트리거(train.py 서브프로세스) →
          재학습 완료(모델 reload) → 추론 재개(resume_s).
    Baseline 은 재학습 동안 추론을 중단하므로 시계열에 드롭 후 재상승이 나타난다.
    """
    if not os.path.exists(config.TRAIN_SCRIPT):
        raise FileNotFoundError(f"TRAIN_SCRIPT not found: {config.TRAIN_SCRIPT}")
    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(f"TRAIN_CSV not found: {config.TRAIN_CSV}")

    inference.get_model()
    win = _dummy_windows(1)[0]
    retrain_epochs = int(os.environ.get('RETRAIN_EPOCHS', '3'))

    sampler = _ResourceSampler(interval_s=1.0)
    sampler.start()
    events = []

    def _serve_for(seconds, phase):
        sampler.set_phase(phase)
        ok = 0
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            try:
                inference.predict_single(win)
                ok += 1
            except Exception as e:                       # noqa: BLE001
                print(f"[retrain-bench] inference error: {e}", file=sys.stderr)
            time.sleep(0.2)
        return ok

    # 1) 정상 추론
    served_before = _serve_for(normal_s, 'normal_before')
    events.append({'event': 'retrain_start', 't': round(
        time.perf_counter() - sampler._t0, 3)})

    # 2) 재학습 트리거 — 추론 중단 (in-process 모델 교체 준비)
    sampler.set_phase('retraining')
    cmd = [sys.executable, config.TRAIN_SCRIPT,
           '--train-csv', config.TRAIN_CSV,
           '--epochs', str(retrain_epochs),
           '--window', str(config.WINDOW_SIZE),
           '--output-dir', os.path.dirname(config.MODEL_PATH)]
    print(f"[retrain-bench] {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    events.append({'event': 'retrain_done', 't': round(
        time.perf_counter() - sampler._t0, 3),
        'returncode': proc.returncode})

    # 3) 모델 reload (재학습 산출물 반영) → 추론 재개
    sampler.set_phase('reloading')
    inference.reload_model()
    served_after = _serve_for(resume_s, 'resume')

    sampler.stop()

    payload = {
        'name': 'retraining',
        'model_name': config.MODEL_NAME,
        'retrain_epochs': retrain_epochs,
        'retrain_returncode': proc.returncode,
        'served_before': served_before,
        'served_after': served_after,
        'events': events,
        'timeseries': sampler.samples,
    }
    if proc.returncode != 0:
        payload['retrain_stderr_tail'] = proc.stderr[-2000:]
    path = _save('retraining', payload)
    payload['results_path'] = path
    return payload
