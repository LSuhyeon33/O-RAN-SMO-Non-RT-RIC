import os

# Baseline rApp: 모델을 프로세스 내부에 in-process로 임베드한다.
# Proposed(ids-rapp-bi-lstm)와 달리 KServe(AIHP) HTTP 호출이 없으므로
# KSERVE_HOST / PREDICT_URL / MODEL_META_URL / MODEL_ZIP_URL 등은 제거하고,
# 이미지에 베이크된 로컬 산출물 경로(MODEL_PATH / SCALER_PATH)만 사용한다.
MODEL_NAME = os.environ.get('MODEL_NAME', 'baseline-bi-lstm')   # /healthz 식별용

# 학습(train.py) 산출물 — Dockerfile이 /model 로 베이크 인.
MODEL_PATH = os.environ.get('MODEL_PATH', '/model/model.keras')
SCALER_PATH = os.environ.get('SCALER_PATH', '/model/scaler.pkl')

INFLUX_URL = os.environ.get('INFLUX_URL', 'http://influxdb2.smo.svc.cluster.local:8086')
INFLUX_ORG = os.environ.get('INFLUX_ORG', 'est')
INFLUX_BUCKET = os.environ.get('INFLUX_BUCKET', 'oom_ids')
INFLUX_MEASUREMENT = os.environ.get('INFLUX_MEASUREMENT', 'oom_test')
INFLUX_TOKEN = os.environ.get('INFLUX_TOKEN')

# 학습 윈도우와 일치(20). Proposed의 기본값 30과 다르므로 명시적으로 둔다.
WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', '30'))

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/results/baseline-bi-lstm')
# /evaluate 동안 시스템 자원(CPU/메모리/GPU)을 샘플링하는 간격(초).
SYSTEM_SAMPLE_INTERVAL_S = float(os.environ.get('SYSTEM_SAMPLE_INTERVAL_S', '2.0'))
HTTP_PORT = int(os.environ.get('HTTP_PORT', '8005'))
REQUEST_TIMEOUT_S = int(os.environ.get('REQUEST_TIMEOUT_S', '60'))
