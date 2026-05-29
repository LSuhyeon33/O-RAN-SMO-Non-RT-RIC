#!/usr/bin/env python
# coding: utf-8
# =============================================================================
# OOM Detection Pipeline — LSTM (multi-class classifier, sliding window)
#
# 입력 : Cassandra feature_store.oom_b_riclog (적재 시점에 ts_ns 정렬·split 완료)
# 출력 : LSTM 모델 (Keras savedmodel) + cv_results.json + training_history.csv/json
# 검증 : Stratified 5-fold 교차검증 (bilstm과 동일) → CV 후 전체 데이터로 최종 재학습
# 주의 : 시퀀스 시간 구조 보존 위해 SMOTE 생략 → class_weight='balanced'로 대체
# =============================================================================

import os
import kfp
import kfp.dsl as dsl
from kfp.dsl import component, Output, Artifact
from kfp import kubernetes


BASE_IMAGE = "traininghost/pipelineimage:gpu-latest"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=['scikit-learn', 'imbalanced-learn', 'nvidia-cuda-nvcc-cu12'],  # nvcc-cu12: GPU runtime 이미지에 없는 ptxas+libdevice 제공
)
def train_export_model(
    featurepath: str,
    modelname: str,
    modelversion: str,
    hidden_size: int = 128,
    num_layers: int = 2,
    window_size: int = 20,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 256,
    n_splits: int = 5,
    seed: int = 42,
    cv_results_output: Output[Artifact] = None,              # ★ KFP가 자동 주입 — cv_results.json
    training_history_csv_output: Output[Artifact] = None,    # ★ KFP가 자동 주입 — training_history.csv
    training_history_json_output: Output[Artifact] = None,   # ★ KFP가 자동 주입 — training_history.json
):
    import os, json, csv, time, random, shutil
    # --- GPU codegen 의존성 경로 설정 (nvidia-cuda-nvcc-cu12 의 ptxas + libdevice.10.bc) ---
    # GPU 이미지(nvidia/cuda:...-runtime)에는 ptxas/libdevice가 없어 Adam의 Pow 연산이 TF MLIR
    # kernel_gen 경로를 탈 때 "JIT compilation failed"로 죽음. nvcc-cu12 가 둘 다 포함.
    import glob, importlib.util
    _spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if _spec and _spec.submodule_search_locations:
        _nvcc_root = list(_spec.submodule_search_locations)[0]
        _ptxas = glob.glob(os.path.join(_nvcc_root, "**", "ptxas"), recursive=True)
        if _ptxas:
            os.environ["PATH"] = os.path.dirname(_ptxas[0]) + os.pathsep + os.environ.get("PATH", "")
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_nvcc_root}"
            print(f"[gpu] ptxas={_ptxas[0]}, XLA_FLAGS={os.environ['XLA_FLAGS']}")
    import numpy as np
    import pandas as pd
    import requests
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)
    from featurestoresdk.feature_store_sdk import FeatureStoreSdk
    from modelmetricsdk.model_metrics_sdk import ModelMetricsSdk

    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

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
    LABEL = 'label'; N_CLASSES = 6; N_FEATURES = len(FEATURES)

    #-- 데이터 로드 (Cassandra 적재 시점에 ts_ns 정렬 완료 → 재정렬 불필요)
    fs_sdk = FeatureStoreSdk()
    print(f"[load] featurepath={featurepath}")
    df = fs_sdk.get_features(featurepath, FEATURES + [LABEL])
    print(f"[load] rows={len(df)}")

    X_df = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y_ser = df[LABEL].astype(int)
    X_arr_all = X_df.values
    y_arr_all = y_ser.values

    #-- 슬라이딩 윈도우 생성 — DL은 SMOTE 생략 (시퀀스 시간 구조 깨짐 방지)
    def make_windows(X, y, w):
        Xs, ys = [], []
        for i in range(len(X) - w + 1):
            Xs.append(X[i:i + w]); ys.append(y[i + w - 1])
        return np.array(Xs), np.array(ys)

    #-- 모델 빌더 (fold마다 새 모델을 만들어야 하므로 함수로 분리)
    def build_model():
        #-- 모델 — stacked LSTM
        m = Sequential()
        for li in range(int(num_layers)):
            return_seq = (li < int(num_layers) - 1)
            if li == 0:
                m.add(LSTM(int(hidden_size), activation='tanh', return_sequences=return_seq,
                           input_shape=(int(window_size), N_FEATURES)))
            else:
                m.add(LSTM(int(hidden_size), activation='tanh', return_sequences=return_seq))
            m.add(Dropout(float(dropout)))
        m.add(Dense(N_CLASSES, activation='softmax'))
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], jit_compile=False)  # XLA auto-JIT 끔
        m.summary()
        return m

    #========================================================================
    #-- Stratified K-fold Cross Validation
    #   슬라이딩 윈도우를 먼저 만든 뒤 윈도우 단위로 계층 분할한다.
    #   StratifiedKFold(shuffle=True)는 각 fold가 6개 클래스를 원래 비율대로
    #   포함하도록 분할 → 클래스가 시간 블록으로 분리된 데이터에서도 정상 평가.
    #========================================================================
    X_seq_full, y_seq_full = make_windows(X_arr_all, y_arr_all, int(window_size))
    print(f"[window] total_seq={X_seq_full.shape}")

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    fold_metrics = []
    fold_val_losses = []
    fold_best_epochs = []
    cm_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)   # fold별 혼동행렬 누적 (= 전체 OOF)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_seq_full, y_seq_full), start=1):
        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx}/{n_splits}] train_seq={len(tr_idx)}, val_seq={len(va_idx)}")
        print(f"{'='*60}")

        X_tr_seq, y_tr_seq = X_seq_full[tr_idx], y_seq_full[tr_idx]
        X_va_seq, y_va_seq = X_seq_full[va_idx], y_seq_full[va_idx]

        # 표준화: 이 fold의 학습 윈도우에만 fit (data leakage 방지).
        #   윈도우 (n, w, feat) → (n*w, feat) 로 펴서 fit/transform 후 원복.
        sc = StandardScaler()
        n_tr, w_len, n_feat = X_tr_seq.shape
        n_va = X_va_seq.shape[0]
        X_tr_seq = sc.fit_transform(X_tr_seq.reshape(-1, n_feat)).reshape(n_tr, w_len, n_feat)
        X_va_seq = sc.transform(X_va_seq.reshape(-1, n_feat)).reshape(n_va, w_len, n_feat)
        print(f"  [window] train_seq={X_tr_seq.shape}, val_seq={X_va_seq.shape}")

        # 클래스 가중치 (이 fold의 학습셋 분포 기준)
        present = np.unique(y_tr_seq)
        cw_vals = compute_class_weight('balanced', classes=present, y=y_tr_seq)
        class_weight = {int(c): float(w) for c, w in zip(present, cw_vals)}
        for c in range(N_CLASSES):
            class_weight.setdefault(c, 1.0)
        print(f"  [class_weight] {class_weight}")

        # 모델 학습 (fold마다 새로 시작)
        model = build_model()
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=int(epochs), batch_size=int(batch_size),
            class_weight=class_weight,
            callbacks=[es],
            verbose=2,
        )

        best_val_loss = float(np.min(history.history['val_loss']))
        best_epoch = int(np.argmin(history.history['val_loss'])) + 1
        fold_val_losses.append(best_val_loss)
        fold_best_epochs.append(best_epoch)

        # fold별 분류 메트릭 (통일 스키마 — 전 메트릭 계산 후 CV 평균)
        y_proba = model.predict(X_va_seq, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        acc = float(accuracy_score(y_va_seq, y_pred))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_va_seq, y_pred, average='macro', zero_division=0)
        f1_w = float(precision_recall_fscore_support(
            y_va_seq, y_pred, average='weighted', zero_division=0)[2])
        cm = confusion_matrix(y_va_seq, y_pred, labels=list(range(N_CLASSES)))
        cm_total += cm                                          # fold 혼동행렬 누적
        fnr_pc = []                                             # 클래스별 FNR = FN / (FN + TP)
        for i in range(N_CLASSES):
            fn = int(cm[i].sum() - cm[i, i]); tp = int(cm[i, i])
            fnr_pc.append(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0)
        fnr_avg = float(np.mean(fnr_pc))
        try:
            auc = float(roc_auc_score(y_va_seq, y_proba, multi_class='ovr',
                                      labels=list(range(N_CLASSES))))
        except Exception as e:
            print(f"  [warn] fold {fold_idx} AUC 계산 실패: {e}"); auc = None

        fold_metrics.append({
            'fold': fold_idx,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'accuracy': acc,
            'f1_macro': float(f1_m),
            'f1_weighted': f1_w,
            'precision_macro': float(p_m),
            'recall_macro': float(r_m),
            'fnr_avg': fnr_avg,
            'fnr_per_class': fnr_pc,
            'roc_auc_ovr': auc,
        })
        print(f"  [fold {fold_idx}] best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}, "
              f"acc={acc:.4f}, f1_macro={f1_m:.4f}, fnr_avg={fnr_avg:.4f}")

        # 메모리 정리 (다음 fold를 위해)
        del model, history
        tf.keras.backend.clear_session()

    #-- CV 결과 집계  ===========================================================
    if not fold_val_losses:
        raise RuntimeError("모든 fold에서 학습이 실패했습니다. window_size나 데이터 길이를 확인하세요.")

    avg_val_loss = float(np.mean(fold_val_losses))
    std_val_loss = float(np.std(fold_val_losses))
    avg_best_epoch = max(1, int(np.round(np.mean(fold_best_epochs))))

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
        print(f"  Fold {fm['fold']}: val_loss={fm['best_val_loss']:.4f} "
              f"(best epoch {fm['best_epoch']}), acc={fm['accuracy']:.4f}, "
              f"f1_macro={fm['f1_macro']:.4f}, fnr_avg={fm['fnr_avg']:.4f}")
    print(f"\n  >>> AVG VAL LOSS: {avg_val_loss:.4f}  (std ±{std_val_loss:.4f})")
    print(f"  >>> AVG acc={avg_accuracy:.4f}, f1_macro={avg_f1_macro:.4f}, "
          f"fnr={avg_fnr:.4f}, auc={'n/a' if avg_roc_auc is None else round(avg_roc_auc, 4)}")
    print(f"  >>> Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"window_size={window_size}, dropout={dropout}, batch_size={batch_size}")
    print(f"{'#'*60}\n")

    out_dir = './'
    # CV 결과 JSON 저장 (실험 추적용 — KFP UI artifact로도 노출)
    cv_summary = {
        'hyperparameters': {
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'window_size': window_size, 'dropout': dropout,
            'batch_size': batch_size, 'epochs_cap': epochs,
            'n_splits': n_splits, 'seed': seed,
        },
        'fold_metrics': fold_metrics,
        'avg_val_loss': avg_val_loss,
        'std_val_loss': std_val_loss,
        'avg_best_epoch': avg_best_epoch,
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
    #-- 최종 모델: 전체 데이터로 재학습
    #   CV에서 얻은 평균 best_epoch만큼 학습 → 과적합/과소적합 회피.
    #   (검증셋 없이 학습하므로 EarlyStopping은 사용하지 않음)
    #========================================================================
    print(f"\n[final] training on ALL data with epochs={avg_best_epoch}")

    sc_final = StandardScaler()
    X_all_scaled = sc_final.fit_transform(X_arr_all)
    X_all_seq, y_all_seq = make_windows(X_all_scaled, y_arr_all, int(window_size))
    print(f"[final] all_seq={X_all_seq.shape}")

    present_all = np.unique(y_all_seq)
    cw_all = compute_class_weight('balanced', classes=present_all, y=y_all_seq)
    class_weight_final = {int(c): float(w) for c, w in zip(present_all, cw_all)}
    for c in range(N_CLASSES):
        class_weight_final.setdefault(c, 1.0)

    model = build_model()
    csv_logger = CSVLogger(os.path.join(out_dir, 'training_history.csv'))
    history_final = model.fit(
        X_all_seq, y_all_seq,
        epochs=avg_best_epoch, batch_size=int(batch_size),
        class_weight=class_weight_final,
        callbacks=[csv_logger],
        verbose=2,
    )

    # history 내용을 JSON 타입으로 변환
    hist_dict = {'epoch': list(range(1, len(history_final.history['loss']) + 1))}
    for k, v in history_final.history.items():
        hist_dict[k] = [float(x) for x in v]
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    #-- training_history 를 KFP UI 아티팩트로도 추출
    if training_history_csv_output is not None:
        shutil.copy(os.path.join(out_dir, 'training_history.csv'), training_history_csv_output.path)
        print(f"[artifact] training_history.csv → {training_history_csv_output.path}")
    if training_history_json_output is not None:
        shutil.copy(os.path.join(out_dir, 'training_history.json'), training_history_json_output.path)
        print(f"[artifact] training_history.json → {training_history_json_output.path}")

    #-- 추론 지연 측정 (최종 모델, 배치 1건 — warmup 100 / 측정 1000회)
    x_one = X_all_seq[:1]
    for _ in range(100):
        model.predict(x_one, verbose=0)
    _lat = []
    for _ in range(1000):
        _t0 = time.perf_counter()
        model.predict(x_one, verbose=0)
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
        'AvgBestEpoch': avg_best_epoch,
        'NSplits': int(n_splits),
        'FoldDetails': fold_metrics,
        'Hyperparameters': cv_summary['hyperparameters'],
    }
    print(f"[eval] {json.dumps(metrics, indent=2)}")

    #-- 모델 export 및 등록
    model.export(out_dir)            # TF SavedModel로 export (Keras 3: model.save는 .keras/.h5 확장자 필요)
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


@dsl.pipeline(name="oom_lstm Pipeline", description="OOM detection — LSTM")
def super_model_pipeline(
    featurepath: str, modelname: str, modelversion: str,
    hidden_size: int = 128, num_layers: int = 2, window_size: int = 20,
    dropout: float = 0.2, epochs: int = 50, batch_size: int = 256,
    n_splits: int = 5, seed: int = 42,
):
    op = train_export_model(
        featurepath=featurepath, modelname=modelname, modelversion=modelversion,
        hidden_size=hidden_size, num_layers=num_layers, window_size=window_size,
        dropout=dropout, epochs=epochs, batch_size=batch_size,
        n_splits=n_splits, seed=seed,
    )
    op.set_caching_options(False)
    kubernetes.set_image_pull_policy(op, "IfNotPresent")
    op.set_accelerator_type("nvidia.com/gpu")          # GPU 1장 요청
    op.set_accelerator_limit(1)
    kubernetes.add_node_selector(op, "gpu", "nvidia")  # GPU 노드(oran-server)에 강제 배치


pipeline_func = super_model_pipeline
file_name = "oom_lstm_pipeline"
kfp.compiler.Compiler().compile(pipeline_func, f"{file_name}.yaml")

import requests
TM_URL = os.environ.get('TM_URL', 'http://tm.traininghost:32002')
pipeline_name = "oom_lstm_Pipeline"
resp = requests.post(f"{TM_URL}/pipelines/{pipeline_name}/upload",
                     files={'file': open(f"{file_name}.yaml", 'rb')})
print(f"[upload] {resp.status_code} {resp.text[:200]}")
