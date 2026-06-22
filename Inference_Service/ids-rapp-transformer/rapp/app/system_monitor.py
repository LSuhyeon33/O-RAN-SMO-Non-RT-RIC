"""추론 중 시스템 자원 사용량 샘플러 — /evaluate 동안 2초 간격으로 기록.

CPU%·RSS(MB)는 rApp 프로세스 기준(psutil)으로 측정한다. 실제 추론 GPU 연산은 별도
KServe 모델 서버 pod에서 일어나고 rApp pod엔 GPU가 없으므로, GPU 컬럼은 best-effort다.
NVML(pynvml)이 초기화되면 노드 GPU 값을 기록하고, 불가하면 빈 값으로 둔다.

baseline-bi-lstm/rapp/app/bench.py 의 _ResourceSampler 패턴을 재사용.
"""
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SystemSampler:
    """백그라운드에서 rApp 프로세스 CPU%·RSS, (가능하면) 노드 GPU 를 일정 간격으로 샘플링."""

    FIELDS = ['timestamp_iso', 't_rel_s', 'phase', 'cpu_percent', 'rss_mb',
              'gpu_util_percent', 'gpu_mem_used_mb']

    def __init__(self, interval_s: float = 2.0):
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.interval = float(interval_s)
        self.samples = []          # [{...FIELDS...}]
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._phase = 'idle'
        self.proc.cpu_percent(None)   # 첫 호출은 0.0 → 기준선 prime

        # --- GPU best-effort 초기화 --------------------------------------- #
        self._gpu = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception as e:
            print(f"[system_monitor] GPU 모니터링 비활성(NVML 초기화 실패): {e}",
                  file=sys.stderr, flush=True)

    def set_phase(self, phase: str):
        self._phase = phase

    def _read_gpu(self):
        """(gpu_util_percent, gpu_mem_used_mb) — 사용 불가 시 (None, None)."""
        if self._gpu is None:
            return None, None
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu
            mem_used = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / (1024 * 1024)
            return float(util), float(mem_used)
        except Exception:
            return None, None

    def _loop(self):
        while not self._stop.is_set():
            cpu = self.proc.cpu_percent(None)
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
            gpu_util, gpu_mem = self._read_gpu()
            self.samples.append({
                'timestamp_iso': _iso_now(),
                't_rel_s': round(time.perf_counter() - self._t0, 3),
                'phase': self._phase,
                'cpu_percent': float(cpu),
                'rss_mb': float(rss_mb),
                'gpu_util_percent': gpu_util,
                'gpu_mem_used_mb': gpu_mem,
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

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'system_samples.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS, restval='')
            w.writeheader()
            for row in self.samples:
                # GPU 값이 None 이면 빈 셀로 출력
                w.writerow({k: ('' if row.get(k) is None else row[k])
                            for k in self.FIELDS})
        print(f"[system_monitor] saved {path} ({len(self.samples)} samples)",
              file=sys.stderr, flush=True)
        return path
