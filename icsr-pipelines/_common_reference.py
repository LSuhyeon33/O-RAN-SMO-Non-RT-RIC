# =============================================================================
# IDS rApp 실험 1 — 6개 모델 KFP 파이프라인 공통 로직 참고용
# =============================================================================
#
# 이 파일은 oom_pipeline_*.py 파일들의 component 함수 안에 인라인 복제되는
# 공통 로직의 "복붙 원본"이다. KFP component는 폐쇄 컨테이너 환경에서 실행되어
# 외부 모듈 import가 어렵기 때문에, 각 파이프라인 파일이 자체적으로 동일한
# 헬퍼를 함수 본문 안에 들고 있다.
#
# 이 파일 자체는 어떤 파이프라인에서도 import되지 않는다.
# 헬퍼를 한 곳에서 보고 일관성을 유지하기 위한 단일 참조점이다.
#
# 참고: /usr/local/o-ran/aimlfw-dep/samples/qoe/qoe_pipeline_l_release.py
# =============================================================================

# OOM 데이터 입력 컬럼 (idx, ts_ns 제외)
OOM_FEATURE_COLUMNS = [
    'inter_arrival_time_ms', 'request_response_latency_ms',
    'timeouts_last_30s', 'timeouts_last_60s', 'total_correlation_errors',
    'errors_last_30s', 'subscription_mismatch', 'subscription_mismatches_last_30s',
    'ssn_retransmission', 'ssn_retransmissions_last_30s',
    'tsn_out_of_order', 'tsn_out_of_order_last_30s',
    'message_size_anomaly', 'msg_size_anomalies_last_30s',
    'consecutive_tx', 'consecutive_rx',
    'arrival_time_cv_30', 'msg_size_cv_30',
    'event_rate_30s', 'tx_rx_ratio_30',
]
LABEL_COLUMN = 'label'
N_CLASSES = 6  # 0=baseline, 1=S1_Normal, 2=S1_Termination, 3=S2, 4=S3, 5=S4


def split_validation(X_df, y_series, n_per_class=100, seed=42):
    """클래스별 n_per_class건을 stratified random sampling → validation, 나머지 → training.

    Cassandra 적재 시점에 ts_ns 정렬 + train/test split이 이미 적용되어
    있으므로 파이프라인은 시간순 정렬이나 80/20 분할을 다시 하지 않는다.
    검증셋만 클래스 균형을 맞춰서 따로 분리한다.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    val_idx = []
    for cls in sorted(y_series.unique()):
        cls_idx = y_series.index[y_series == cls].to_numpy()
        if len(cls_idx) <= n_per_class:
            sampled = cls_idx
        else:
            sampled = rng.choice(cls_idx, size=n_per_class, replace=False)
        val_idx.extend(sampled.tolist())
    val_idx_set = set(val_idx)
    train_idx = [i for i in X_df.index if i not in val_idx_set]
    val_idx_sorted = sorted(val_idx)
    return (
        X_df.loc[train_idx].reset_index(drop=True),
        X_df.loc[val_idx_sorted].reset_index(drop=True),
        y_series.loc[train_idx].reset_index(drop=True),
        y_series.loc[val_idx_sorted].reset_index(drop=True),
    )


def make_windows(X_arr, y_arr, window):
    """Sliding window 변환. 출력 shape = (N - window + 1, window, n_features),
    각 윈도우의 레이블은 마지막 위치의 레이블."""
    import numpy as np
    Xs, ys = [], []
    for i in range(len(X_arr) - window + 1):
        Xs.append(X_arr[i:i + window])
        ys.append(y_arr[i + window - 1])
    return np.array(Xs), np.array(ys)


def evaluate_classification(y_true, y_pred, y_proba=None, n_classes=N_CLASSES):
    """분류 평가 메트릭 — Accuracy / F1(macro,weighted) / FNR(per-class+avg) / ROC-AUC(OvR)."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fnr_per_class = []
    for i in range(n_classes):
        fn = int(cm[i].sum() - cm[i, i])
        tp = int(cm[i, i])
        denom = fn + tp
        fnr_per_class.append(float(fn / denom) if denom > 0 else 0.0)
    fnr_avg = float(np.mean(fnr_per_class))
    auc_ovr = None
    if y_proba is not None:
        try:
            present = sorted(set(int(x) for x in y_true))
            if len(present) == n_classes:
                auc_ovr = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr',
                    labels=list(range(n_classes))))
        except Exception as e:
            print(f"[warn] ROC-AUC computation failed: {e}")
    return {
        'Accuracy': float(acc),
        'F1_macro': float(f1_m),
        'F1_weighted': float(f1_w),
        'Precision_macro': float(p_m),
        'Recall_macro': float(r_m),
        'FNR_avg': fnr_avg,
        'FNR_per_class': fnr_per_class,
        'ROC_AUC_ovr': auc_ovr,
        'ConfusionMatrix': cm.tolist(),
    }


def measure_inference_latency(predict_fn, x_one_sample, n_iter=1000, warmup=100):
    """배치 1건 추론 지연 측정.
    predict_fn(x) → 추론 호출. x_one_sample은 (1, ...) shape의 단일 샘플.
    반환: (mean_ms, std_ms)."""
    import time
    import numpy as np
    for _ in range(warmup):
        predict_fn(x_one_sample)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(x_one_sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def save_training_history(history_dict, out_dir='./'):
    """epoch별 학습 곡선을 CSV + JSON 두 형식으로 저장.
    history_dict 형태: {'epoch':[...], 'loss':[...], 'val_loss':[...], 'accuracy':[...], 'val_accuracy':[...]}
    또는 keras history.history 그대로 (epoch 키 자동 추가).
    저장 위치는 모델 디렉터리 → mm_sdk.upload_model로 함께 업로드되어 다운로드 가능.
    """
    import json, csv, os
    if 'epoch' not in history_dict:
        any_metric = next(iter(history_dict.values()))
        history_dict = {'epoch': list(range(1, len(any_metric) + 1)), **history_dict}
    json_path = os.path.join(out_dir, 'training_history.json')
    csv_path = os.path.join(out_dir, 'training_history.csv')
    with open(json_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    keys = list(history_dict.keys())
    n_rows = len(history_dict[keys[0]])
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n_rows):
            w.writerow([history_dict[k][i] for k in keys])
    print(f"[history] saved: {csv_path}, {json_path}")


# -----------------------------------------------------------------------------
# 모델 등록부 / 메트릭 업로드 — 모든 파이프라인 공통 후처리 패턴
# -----------------------------------------------------------------------------
def register_and_upload(out_dir, modelname, modelversion, featurepath,
                         metrics_dict, mm_sdk):
    """학습 완료 후 모델 등록부 갱신 + 메트릭/모델 업로드.
    qoe_pipeline_l_release.py와 동일 패턴."""
    import requests, json
    artifactversion = '1.0.0'
    url = ("http://modelmgmtservice.traininghost:8082"
           f"/ai-ml-model-registration/v1/model-registrations/updateArtifact"
           f"/{modelname}/{modelversion}/{artifactversion}")
    resp = requests.post(url).json()
    print(f"[mms] updated: {resp}")

    trainingjob_id = featurepath.split('_')[-1]
    payload = {'metrics': [metrics_dict]}
    mm_sdk.upload_metrics(payload, trainingjob_id)
    print(f"[metrics] uploaded for jobid={trainingjob_id}: "
          f"{mm_sdk.get_metrics(trainingjob_id)}")
    mm_sdk.upload_model(out_dir, modelname, modelversion, artifactversion)
    print(f"[model] uploaded: {modelname}/{modelversion}/{artifactversion}")
