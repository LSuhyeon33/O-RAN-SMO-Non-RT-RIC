"""커스텀 KServe V1 predictor — 트리 모델(sklearn joblib / XGBoost json) 서빙.

배포된 InferenceService 의 Model.zip 번들을 받아 모델 파일을 로드하고 predict_proba 로
6-class 확률을 반환한다. 두 포맷을 MODEL_FILENAME 확장자로 구분한다:
  - *.joblib            : joblib.load (RandomForest 등 sklearn)
  - *.json (xgboost)    : xgboost.XGBClassifier().load_model
스케일링은 호출자(rApp)가 이미 번들의 scaler.pkl 로 수행하므로 여기서는 다시 정규화하지 않는다.

메모리: RandomForest(300 trees, depth 무제한)는 로드만으로 ~4GB RSS 가 필요하다(XGBoost 는 작음).
번들은 디스크로 스트리밍 후 임시파일에서 로드하여 메모리 피크를 줄인다.
모델 로드는 락으로 1회만 수행한다(readiness probe 가 동시에 여러 다운로드를 일으키지 않도록).
"""
import os
import shutil
import sys
import tempfile
import threading
import zipfile

import joblib
import numpy as np
import requests
from flask import Flask, request, jsonify

MODEL_NAME = os.environ.get('MODEL_NAME', 'rf-riclog')
MODEL_ZIP_URL = os.environ.get(
    'MODEL_ZIP_URL',
    'http://210.123.36.94:32002/model/rf-04/1/1.0.0/Model.zip')
MODEL_FILENAME = os.environ.get('MODEL_FILENAME', 'rf_model.joblib')
PORT = int(os.environ.get('PORT', '8080'))
REQUEST_TIMEOUT_S = int(os.environ.get('REQUEST_TIMEOUT_S', '300'))

app = Flask(__name__)
_MODEL = None
_LOCK = threading.Lock()


def _load_model():
    """Model.zip 을 디스크로 스트리밍 → 모델 항목을 임시파일로 추출 → joblib.load.
    번들 내 경로 구조(1/ 등)와 무관하게 basename 으로 매칭한다.
    """
    print(f"[model] downloading bundle: {MODEL_ZIP_URL}", file=sys.stderr, flush=True)
    with tempfile.NamedTemporaryFile(suffix='.zip') as tmp_zip:
        with requests.get(MODEL_ZIP_URL, stream=True, timeout=REQUEST_TIMEOUT_S) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp_zip.write(chunk)
        tmp_zip.flush()
        with zipfile.ZipFile(tmp_zip.name) as zf:
            matches = [n for n in zf.namelist()
                       if n.rsplit('/', 1)[-1] == MODEL_FILENAME]
            if not matches:
                raise FileNotFoundError(
                    f"'{MODEL_FILENAME}' not in bundle; entries={zf.namelist()}")
            suffix = '.json' if MODEL_FILENAME.endswith('.json') else '.joblib'
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_model:
                with zf.open(matches[0]) as src:
                    shutil.copyfileobj(src, tmp_model)
                tmp_model.flush()
                if suffix == '.json':
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(tmp_model.name)
                else:
                    model = joblib.load(tmp_model.name)
    print(f"[model] loaded '{matches[0]}' -> {type(model).__name__}, "
          f"classes_={getattr(model, 'classes_', None)}", file=sys.stderr, flush=True)
    return model


def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _MODEL = _load_model()
    return _MODEL


@app.get(f'/v1/models/{MODEL_NAME}')
def model_metadata():
    # readiness probe + 호출자(kserve_reachable) 용. 모델 로드 성공해야 ready=True.
    try:
        get_model()
        return jsonify({'name': MODEL_NAME, 'ready': True})
    except Exception as e:
        print(f"[meta] model not ready: {e}", file=sys.stderr, flush=True)
        return jsonify({'name': MODEL_NAME, 'ready': False, 'error': str(e)}), 503


@app.post(f'/v1/models/{MODEL_NAME}:predict')
def predict():
    body = request.get_json(force=True)
    instances = body.get('instances')
    if instances is None:
        return jsonify({'error': 'body must contain "instances"'}), 400
    X = np.asarray(instances, dtype=float)
    if X.ndim != 2:
        return jsonify({'error': f'instances must be 2-D (N, n_features), got shape {X.shape}'}), 400
    probs = get_model().predict_proba(X)
    return jsonify({'predictions': probs.tolist()})


if __name__ == '__main__':
    try:
        get_model()
        print("[startup] model loaded", file=sys.stderr, flush=True)
    except Exception as e:
        # 기동 실패해도 프로세스는 살려 두고 readiness 가 503 을 내도록 한다(로그 확인 용이).
        print(f"[startup] model load failed: {e}", file=sys.stderr, flush=True)
    app.run(host='0.0.0.0', port=PORT, threaded=True)
