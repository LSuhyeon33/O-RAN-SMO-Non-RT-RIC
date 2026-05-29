import os

KSERVE_HOST = os.environ.get('KSERVE_HOST', 'http://bi-lstm-01.kserve-test.svc.cluster.local')
MODEL_NAME = os.environ.get('MODEL_NAME', 'bi-lstm-01')
PREDICT_URL = f"{KSERVE_HOST}/v1/models/{MODEL_NAME}:predict"
MODEL_META_URL = f"{KSERVE_HOST}/v1/models/{MODEL_NAME}"

INFLUX_URL = os.environ.get('INFLUX_URL', 'http://influxdb2.smo.svc.cluster.local:8086')
INFLUX_ORG = os.environ.get('INFLUX_ORG', 'est')
INFLUX_BUCKET = os.environ.get('INFLUX_BUCKET', 'oom_ids')
INFLUX_MEASUREMENT = os.environ.get('INFLUX_MEASUREMENT', 'oom_test')
INFLUX_TOKEN = os.environ.get('INFLUX_TOKEN')

WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', '10'))

# --- scaler 위치 -------------------------------------------------------------
# 기본: 모델 번들(Model.zip)에서 받아 쓴다 → KServe가 서빙하는 모델과 버전 일치 보장.
# MODEL_ZIP_URL 은 KServe InferenceService 의 storageUri 와 동일 값으로 둔다.
MODEL_ZIP_URL = os.environ.get(
    'MODEL_ZIP_URL',
    'http://210.123.36.94:32002/model/bi-lstm-01/1/1.0.0/Model.zip')
SCALER_FILENAME = os.environ.get('SCALER_FILENAME', 'scaler.pkl')
SCALER_CACHE_PATH = os.environ.get('SCALER_CACHE_PATH', '/tmp/scaler.pkl')
# SCALER_PATH 에 값을 주면 번들 다운로드를 건너뛰고 그 파일을 쓴다 (오프라인/테스트용).
SCALER_PATH = os.environ.get('SCALER_PATH', '')
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/results/bi-lstm-01')
HTTP_PORT = int(os.environ.get('HTTP_PORT', '8005'))
REQUEST_TIMEOUT_S = int(os.environ.get('REQUEST_TIMEOUT_S', '60'))
