#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# OOM Detection Pipeline — Random Forest (multi-class classifier)
#
# 입력 : Cassandra feature_store.oom_b_riclog (적재 시점에 ts_ns 정렬·split 완료)
# 출력 : RF 모델 (joblib) + cv_results.json + training_history (fold별 CV 결과)
# 검증 : Stratified 5-fold 교차검증 (bilstm과 동일) → CV 후 전체 데이터로 최종 재학습.
# =============================================================================

import os
import kfp
import kfp.dsl as dsl
from kfp.dsl import component, Output, Artifact
from kfp import kubernetes


BASE_IMAGE = "traininghost/pipelineimage:latest"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=['scikit-learn', 'imbalanced-learn', 'joblib'],
)
def train_export_model(
    featurepath: str,
    modelname: str,
    modelversion: str,
    n_estimators: int = 300,
    max_depth: str = 'None',     # 'None' / '10' / '20' (string으로 받아 변환)
    n_splits: int = 5,
    seed: int = 42,
    cv_results_output: Output[Artifact] = None,              # ★ KFP가 자동 주입 — cv_results.json
    training_history_csv_output: Output[Artifact] = None,    # ★ KFP가 자동 주입 — training_history.csv
    training_history_json_output: Output[Artifact] = None,   # ★ KFP가 자동 주입 — training_history.json
):
    import os, json, csv, time, random, shutil
    import numpy as np
    import pandas as pd
    import requests
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score, log_loss)
    from imblearn.over_sampling import SMOTE
    from featurestoresdk.feature_store_sdk import FeatureStoreSdk
    from modelmetricsdk.model_metrics_sdk import ModelMetricsSdk

    random.seed(seed); np.random.seed(seed)

    FEATURES = [
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
    LABEL = 'label'; N_CLASSES = 6

    md = None if str(max_depth).lower() in ('none', '', 'null') else int(max_depth)

    #-- 데이터 로드 (Cassandra 적재 시점에 ts_ns 정렬 완료 → 재정렬 불필요)
    fs_sdk = FeatureStoreSdk()
    print(f"[load] featurepath={featurepath}")
    df = fs_sdk.get_features(featurepath, FEATURES + [LABEL])
    print(f"[load] rows={len(df)}")

    X_df = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y_ser = df[LABEL].astype(int)
    X_arr_all = X_df.values
    y_arr_all = y_ser.values

    #-- 모델 빌더 (fold마다 새 모델을 만들어야 하므로 함수로 분리)
    def build_model():
        return RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=md,
            class_weight='balanced',
            random_state=seed, n_jobs=-1,
        )

    #========================================================================
    #-- Stratified K-fold Cross Validation
    #   원본 행 단위로 계층 분할한다. fold마다 SMOTE → StandardScaler → fit 을
    #   학습셋 내부에서만 수행한다 (data leakage 방지).
    #========================================================================
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    fold_metrics = []
    fold_val_losses = []
    cm_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)   # fold별 혼동행렬 누적 (= 전체 OOF)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_arr_all, y_arr_all), start=1):
        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx}/{n_splits}] train={len(tr_idx)}, val={len(va_idx)}")
        print(f"{'='*60}")

        X_tr, y_tr = X_arr_all[tr_idx], y_arr_all[tr_idx]
        X_va, y_va = X_arr_all[va_idx], y_arr_all[va_idx]

        # SMOTE: 이 fold의 학습셋에만 적용 (data leakage 방지)
        sm = SMOTE(random_state=seed)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        print(f"  [SMOTE] train rows {len(X_tr)} → {len(X_tr_res)}")

        # 표준화: 이 fold의 (리샘플된) 학습셋에만 fit
        sc = StandardScaler()
        X_tr_res = sc.fit_transform(X_tr_res)
        X_va_s = sc.transform(X_va)

        model = build_model()
        model.fit(X_tr_res, y_tr_res)

        # fold별 분류 메트릭 (통일 스키마 — 전 메트릭 계산 후 CV 평균)
        y_pred = model.predict(X_va_s)
        y_proba = model.predict_proba(X_va_s)
        val_loss = float(log_loss(y_va, y_proba, labels=list(range(N_CLASSES))))
        fold_val_losses.append(val_loss)

        acc = float(accuracy_score(y_va, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_va, y_pred, average='macro', zero_division=0)
        f1_w = float(precision_recall_fscore_support(
            y_va, y_pred, average='weighted', zero_division=0)[2])
        cm = confusion_matrix(y_va, y_pred, labels=list(range(N_CLASSES)))
        cm_total += cm                                          # fold 혼동행렬 누적
        fnr_pc = []                                             # 클래스별 FNR = FN / (FN + TP)
        for i in range(N_CLASSES):
            fn = int(cm[i].sum() - cm[i, i]); tp = int(cm[i, i])
            fnr_pc.append(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0)
        fnr_avg = float(np.mean(fnr_pc))
        try:
            auc = float(roc_auc_score(y_va, y_proba, multi_class='ovr',
                                      labels=list(range(N_CLASSES))))
        except Exception as e:
            print(f"  [warn] fold {fold_idx} AUC 계산 실패: {e}"); auc = None

        fold_metrics.append({
            'fold': fold_idx,
            'val_loss': val_loss,
            'accuracy': acc,
            'f1_macro': float(f1_m),
            'f1_weighted': f1_w,
            'precision_macro': float(p_m),
            'recall_macro': float(r_m),
            'fnr_avg': fnr_avg,
            'fnr_per_class': fnr_pc,
            'roc_auc_ovr': auc,
        })
        print(f"  [fold {fold_idx}] val_loss={val_loss:.4f}, acc={acc:.4f}, "
              f"f1_macro={f1_m:.4f}, fnr_avg={fnr_avg:.4f}")

    #-- CV 결과 집계  ===========================================================
    if not fold_val_losses:
        raise RuntimeError("모든 fold에서 학습이 실패했습니다. n_splits나 데이터를 확인하세요.")

    avg_val_loss = float(np.mean(fold_val_losses))
    std_val_loss = float(np.std(fold_val_losses))

    #-- fold별 메트릭을 평균 (ConfusionMatrix만 fold 합산 = 전체 OOF)
    def _mean(key):
        return float(np.mean([fm[key] for fm in fold_metrics]))
    avg_accuracy = _mean('accuracy')
    avg_f1_macro = _mean('f1_macro')
    avg_f1_weighted = _mean('f1_weighted')
    avg_precision_macro = _mean('precision_macro')
    avg_recall_macro = _mean('recall_macro')
    avg_fnr = _mean('fnr_avg')
    avg_fnr_per_class = [float(np.mean([fm['fnr_per_class'][i] for fm in fold_metrics]))
                         for i in range(N_CLASSES)]
    _aucs = [fm['roc_auc_ovr'] for fm in fold_metrics if fm['roc_auc_ovr'] is not None]
    avg_roc_auc = float(np.mean(_aucs)) if _aucs else None

    print(f"\n{'#'*60}")
    print(f"# Stratified {n_splits}-fold CV Results")
    print(f"{'#'*60}")
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']}: val_loss={fm['val_loss']:.4f}, "
              f"acc={fm['accuracy']:.4f}, f1_macro={fm['f1_macro']:.4f}, "
              f"fnr_avg={fm['fnr_avg']:.4f}")
    print(f"\n  >>> AVG VAL LOSS: {avg_val_loss:.4f}  (std ±{std_val_loss:.4f})")
    print(f"  >>> AVG acc={avg_accuracy:.4f}, f1_macro={avg_f1_macro:.4f}, "
          f"fnr={avg_fnr:.4f}, auc={'n/a' if avg_roc_auc is None else round(avg_roc_auc, 4)}")
    print(f"  >>> Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"{'#'*60}\n")

    out_dir = './'
    # CV 결과 JSON 저장 (실험 추적용 — KFP UI artifact로도 노출)
    cv_summary = {
        'hyperparameters': {
            'n_estimators': n_estimators, 'max_depth': max_depth,
            'n_splits': n_splits, 'seed': seed,
        },
        'fold_metrics': fold_metrics,
        'avg_val_loss': avg_val_loss,
        'std_val_loss': std_val_loss,
        'avg_accuracy': avg_accuracy,
        'avg_f1_macro': avg_f1_macro,
        'avg_f1_weighted': avg_f1_weighted,
        'avg_precision_macro': avg_precision_macro,
        'avg_recall_macro': avg_recall_macro,
        'avg_fnr': avg_fnr,
        'avg_fnr_per_class': avg_fnr_per_class,
        'avg_roc_auc_ovr': avg_roc_auc,
        'confusion_matrix': cm_total.tolist(),
    }
    with open(os.path.join(out_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"[cv] saved cv_results.json")
    if cv_results_output is not None:                          # KFP UI 아티팩트로도 추출
        shutil.copy(os.path.join(out_dir, 'cv_results.json'), cv_results_output.path)
        print(f"[artifact] cv_results.json → {cv_results_output.path}")

    #========================================================================
    #-- 최종 모델: 전체 데이터로 재학습 (전체 SMOTE → StandardScaler → fit)
    #========================================================================
    print(f"\n[final] training on ALL data")
    sm_final = SMOTE(random_state=seed)
    X_all_res, y_all_res = sm_final.fit_resample(X_arr_all, y_arr_all)
    print(f"[final] SMOTE rows {len(X_arr_all)} → {len(X_all_res)}")
    sc_final = StandardScaler()
    X_all_res = sc_final.fit_transform(X_all_res)
    model = build_model()
    model.fit(X_all_res, y_all_res)

    #-- training_history: RF는 epoch 개념 없음 → fold별 CV 결과를 기록
    history = {
        'fold': [fm['fold'] for fm in fold_metrics],
        'val_logloss': [fm['val_loss'] for fm in fold_metrics],
        'val_accuracy': [fm['accuracy'] for fm in fold_metrics],
    }
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, 'training_history.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(history.keys())
        for i in range(len(history['fold'])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"[history] saved {len(history['fold'])} fold entries")

    #-- training_history 를 KFP UI 아티팩트로도 추출
    if training_history_csv_output is not None:
        shutil.copy(os.path.join(out_dir, 'training_history.csv'), training_history_csv_output.path)
        print(f"[artifact] training_history.csv → {training_history_csv_output.path}")
    if training_history_json_output is not None:
        shutil.copy(os.path.join(out_dir, 'training_history.json'), training_history_json_output.path)
        print(f"[artifact] training_history.json → {training_history_json_output.path}")

    #-- 추론 지연 측정 (최종 모델, 배치 1건 — warmup 100 / 측정 1000회)
    x_one = X_all_res[:1]
    for _ in range(100):
        model.predict(x_one)
    _lat = []
    for _ in range(1000):
        _t0 = time.perf_counter()
        model.predict(x_one)
        _lat.append((time.perf_counter() - _t0) * 1000.0)
    lat_mean, lat_std = float(np.mean(_lat)), float(np.std(_lat))
    print(f"[latency] mean={lat_mean:.4f}ms, std={lat_std:.4f}ms")

    #-- 업로드용 메트릭: CV 평균 + fold 합산 혼동행렬 ==========================
    # 모든 메트릭(FNR/AUC/혼동행렬 포함)을 fold별로 계산해 평균한다. StratifiedKFold가
    # fold마다 클래스 분포를 보존하므로 fold 평균은 안정적이며, ConfusionMatrix는 fold가
    # 표본을 분할하므로 합산 = 전체 OOF(out-of-fold) 행렬이 된다.
    metrics = {
        'AvgValLoss': avg_val_loss,                                # ★ 하이퍼파라미터 선정 기준
        'StdValLoss': std_val_loss,
        'FoldValLosses': fold_val_losses,
        'AvgAccuracy': avg_accuracy,
        'AvgF1Macro': avg_f1_macro,
        'AvgF1Weighted': avg_f1_weighted,
        'AvgPrecisionMacro': avg_precision_macro,
        'AvgRecallMacro': avg_recall_macro,
        'AvgFNR': avg_fnr,                                         # ★ 보안 핵심 지표
        'AvgFNRPerClass': avg_fnr_per_class,
        'AvgROCAUCovr': avg_roc_auc,
        'ConfusionMatrix': cm_total.tolist(),
        'Latency_ms_mean': lat_mean,
        'Latency_ms_std': lat_std,
        'NSplits': int(n_splits),
        'FoldDetails': fold_metrics,
        'Hyperparameters': cv_summary['hyperparameters'],
    }
    print(f"[eval] {json.dumps(metrics, indent=2)}")

    #-- 모델 저장 + 등록
    joblib.dump(model, os.path.join(out_dir, 'rf_model.joblib'))
    artifactversion = '1.0.0'
    mms_url = ("http://modelmgmtservice.traininghost:8082"
               f"/ai-ml-model-registration/v1/model-registrations/updateArtifact"
               f"/{modelname}/{modelversion}/{artifactversion}")
    print(f"[mms] {requests.post(mms_url).json()}")
    mm_sdk = ModelMetricsSdk()
    trainingjob_id = featurepath.split('_')[-1]   # 규약: featurepath = <feature_group>_<숫자 trainingjob_id>
    # tm을 거치지 않고 KFP UI에서 직접 실행하면 trainingjob_id가 문자열이 됨 → tm의 update-model-metrics는
    # 정수 id만 받으므로 500. 그 경우 tm 업로드는 건너뛰고 메트릭은 로그로만 남긴다.
    if str(trainingjob_id).isdigit():
        mm_sdk.upload_metrics({'metrics': [metrics]}, trainingjob_id)
        print(f"[metrics] {mm_sdk.get_metrics(trainingjob_id)}")
    else:
        print(f"[metrics] trainingjob_id='{trainingjob_id}' is not numeric → skip tm upload. metrics={json.dumps(metrics)}")
    mm_sdk.upload_model(out_dir, modelname, modelversion, artifactversion)
    print(f"[model] uploaded {modelname}/{modelversion}/{artifactversion}")


@dsl.pipeline(name="oom_rf Pipeline", description="OOM detection — Random Forest")
def super_model_pipeline(
    featurepath: str, modelname: str, modelversion: str,
    n_estimators: int = 300, max_depth: str = 'None',
    n_splits: int = 5, seed: int = 42,
):
    op = train_export_model(
        featurepath=featurepath, modelname=modelname, modelversion=modelversion,
        n_estimators=n_estimators, max_depth=max_depth,
        n_splits=n_splits, seed=seed,
    )
    op.set_caching_options(False)
    kubernetes.set_image_pull_policy(op, "IfNotPresent")


pipeline_func = super_model_pipeline
file_name = "oom_rf_pipeline"
kfp.compiler.Compiler().compile(pipeline_func, f"{file_name}.yaml")

import requests
TM_URL = os.environ.get('TM_URL', 'http://tm.traininghost:32002')
pipeline_name = "oom_rf_Pipeline"
resp = requests.post(f"{TM_URL}/pipelines/{pipeline_name}/upload",
                     files={'file': open(f"{file_name}.yaml", 'rb')})
print(f"[upload] {resp.status_code} {resp.text[:200]}")
