#!/usr/bin/env python
# coding: utf-8

# # OOM Detection Pipeline — Bi-LSTM (multi-class classifier, sliding window)
# 구조 외 모든 부분은 oom_pipeline_lstm.py와 동일.

# ## 1. 라이브러리 임포트

import os
import kfp
import kfp.dsl as dsl
from kfp.dsl import component, Output, Artifact
from kfp import kubernetes


BASE_IMAGE = "traininghost/pipelineimage:gpu-latest"

# ## 2. 학습 컴포넌트 정의 (KFP `@component`)
#
# > ⚠️ 이 셀은 한 덩어리로 유지하세요. `@component` 데코레이터가 붙은 함수 본문을 쪼개면 KFP 컴파일이 깨집니다.

@component(
    base_image=BASE_IMAGE,
    packages_to_install=['scikit-learn', 'imbalanced-learn', 'nvidia-cuda-nvcc-cu12'],  # nvcc-cu12: GPU runtime 이미지에 없는 ptxas+libdevice 제공
)
def train_export_model(
    featurepath: str,                              # Feature Store에서 학습 데이터를 가져올 경로
    modelname: str,                                # 등록할 모델 이름
    modelversion: str,                             # 모델 버전 문자열
    hidden_size: int = 128,                        # LSTM 셀의 은닉 상태 차원 (양방향이므로 실제 출력은 256)
    num_layers: int = 2,                           # Bi-LSTM 층 개수 (스택 깊이)
    window_size: int = 20,                         # 슬라이딩 윈도우 길이 = 한 번에 볼 시퀀스 타임스텝 수
    dropout: float = 0.2,                          # 드롭아웃 비율 (과적합 방지, 20%)
    epochs: int = 100,                             # 최대 학습 에폭 수 (EarlyStopping으로 조기 종료 가능)
    batch_size: int = 256,                         # 미니배치 크기
    n_splits: int = 5,                             # Stratified K-fold CV fold 개수
    seed: int = 42,                                # 난수 시드 (재현성 보장)
    scaler_output: Output[Artifact] = None,                  # ★ KFP가 자동 주입 — 최종 StandardScaler
    cv_results_output: Output[Artifact] = None,              # ★ KFP가 자동 주입 — cv_results.json
    training_history_csv_output: Output[Artifact] = None,    # ★ KFP가 자동 주입 — training_history.csv
    training_history_json_output: Output[Artifact] = None,   # ★ KFP가 자동 주입 — training_history.json
):
    import os, json, csv, time, random, shutil
    # --- GPU codegen 의존성 경로 설정 (nvidia-cuda-nvcc-cu12 pip 패키지의 ptxas + libdevice) ---
    # GPU 이미지는 nvidia/cuda:...-runtime 기반이라 ptxas / libdevice.10.bc 가 없음 →
    # Adam의 Pow 연산이 TF MLIR kernel_gen 경로를 탈 때 "JIT compilation failed" 로 죽음.
    # nvcc-cu12 가 둘 다 포함하므로 PATH(ptxas) + XLA_FLAGS(libdevice 디렉터리)만 잡아주면 됨.
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
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)
    from featurestoresdk.feature_store_sdk import FeatureStoreSdk
    from modelmetricsdk.model_metrics_sdk import ModelMetricsSdk

    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)         # 3대 난수 시드 모두 고정 → 재현성

    #-- 피처 정의
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
    TS_COL = 'ts_ns'                               # 시간 정렬 전용 컬럼 (모델 입력 X)
    LABEL = 'label'                                # 정답 라벨 컬럼명
    N_CLASSES = 6                                  # 분류할 클래스 개수 (정상 + 5가지 OOM 유형 추정)
    N_FEATURES = len(FEATURES)                     # 피처 개수 (20)

    #-- 데이터 로드
    fs_sdk = FeatureStoreSdk()                                              # Feature Store SDK 인스턴스 생성
    print(f"[load] featurepath={featurepath}")                              # 로드 경로 로깅
    df = fs_sdk.get_features(featurepath, FEATURES + [TS_COL, LABEL])       # 피처 + ts_ns + label 컬럼만 가져옴
    print(f"[load] rows={len(df)}")                                         # 가져온 행 수 출력

    #-- 시간 순서대로 정렬 (슬라이딩 윈도우가 시간 연속 시퀀스가 되도록)
    df[TS_COL] = pd.to_numeric(df[TS_COL], errors='coerce')                         # 숫자 변환 (ns 단위 정수 가정)
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)      # ts_ns NaN 제거 + 오름차순 정렬
    print(f"[sort] sorted by {TS_COL}, range=[{int(df[TS_COL].min())}, {int(df[TS_COL].max())}], rows={len(df)}")

    X_df = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)   # 숫자 변환 실패는 NaN→0으로 처리
    y_ser = df[LABEL].astype(int)                                           # 라벨을 정수형으로 강제 변환
    X_arr_all = X_df.values                                                 # (N, N_FEATURES)
    y_arr_all = y_ser.values                                                # (N,)

    #-- 슬라이딩 윈도우 생성
    def make_windows(X, y, w):                     # 시퀀스 윈도우 변환 함수
        Xs, ys = [], []
        for i in range(len(X) - w + 1):            # 길이 w로 한 칸씩 슬라이딩
            Xs.append(X[i:i + w])                  # 입력: w개의 연속 타임스텝
            ys.append(y[i + w - 1])                # 라벨: 윈도우의 마지막 시점 라벨 (many-to-one)
        return np.array(Xs), np.array(ys)

    #-- 모델 빌더 (fold마다 새 모델을 만들어야 하므로 함수로 분리)
    def build_model():
        #-- 모델 — Bidirectional LSTM
        m = Sequential()                           # 순차 모델 생성
        for li in range(int(num_layers)):              # num_layers만큼 층 쌓기
            return_seq = (li < int(num_layers) - 1)    # 마지막 층만 False (벡터 반환), 나머지는 시퀀스 반환
            if li == 0:                                # 첫 번째 층은 input_shape 지정 필요
                m.add(Bidirectional(
                    LSTM(int(hidden_size), activation='tanh', return_sequences=return_seq),
                    input_shape=(int(window_size), N_FEATURES)))     # (윈도우길이, 피처수)
            else:                                      # 두 번째 층부터는 input_shape 자동 추론
                m.add(Bidirectional(
                    LSTM(int(hidden_size), activation='tanh', return_sequences=return_seq)))
            m.add(Dropout(float(dropout)))         # 각 Bi-LSTM 층 뒤에 드롭아웃
        m.add(Dense(N_CLASSES, activation='softmax'))           # 출력층: 6개 클래스 확률
        m.compile(optimizer='adam',                             # Adam 옵티마이저
                  loss='sparse_categorical_crossentropy',       # 라벨이 정수일 때 쓰는 손실 (원-핫 변환 불필요)
                  metrics=['accuracy'],
                  jit_compile=False)                            # XLA JIT 끔 (Keras 3 기본 'auto'면 GPU에서 ptxas 필요 → 이미지에 없음)

        m.summary()                                             # 모델 구조 출력

        return m

    #========================================================================
    #-- Stratified K-fold Cross Validation
    #   StratifiedKFold(shuffle=True)는 각 fold의 학습/검증셋이 모두 6개 클래스를
    #   원래 비율대로 포함하도록 분할한다. 클래스가 시간 블록 단위로 분리된
    #   oom_b_riclog 데이터에서도 학습/검증 분포가 일치 → 정상 평가가 가능하다.
    #   (TimeSeriesSplit은 fold마다 학습/검증 클래스가 어긋나 정확도가 0이 됐음)
    #   슬라이딩 윈도우를 먼저 만든 뒤, 윈도우 단위로 계층 분할한다.
    #========================================================================
    # 전체 데이터를 윈도우로 변환 (분할 전 1회) — 윈도우 라벨 기준으로 계층화
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
        # val_loss가 5에폭 동안 개선 없으면 중단 + 가장 좋았던 가중치 복원
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=int(epochs), batch_size=int(batch_size),
            class_weight=class_weight,
            callbacks=[es],
            verbose=2
        )

        # 이 fold의 최적 val_loss (history의 최솟값 = restore_best_weights가 복원한 시점)
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
        verbose=2
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

    #========================================================================
    #-- 최종 StandardScaler 외부 추출 (파드 종료 후에도 회수 가능하도록)
    #   1) KFP Output[Artifact] → KFP UI에서 직접 다운로드 (메인 경로)
    #   2) out_dir/scaler.pkl   → mm_sdk가 디렉터리 업로드 시 모델과 함께
    #   3) scaler_params.json   → sklearn 버전 무관하게 mean/scale 복원 가능
    #========================================================================
    import joblib
    # (1) KFP 아티팩트로 추출 — 파이프라인 완료 후 UI에서 다운로드
    if scaler_output is not None:                              # 직접 호출 방어 (KFP 외부 실행 대비)
        joblib.dump(sc_final, scaler_output.path)
        print(f"[scaler] KFP artifact saved → {scaler_output.path} "
              f"(URI={getattr(scaler_output, 'uri', 'n/a')})")

    # (2) 모델 디렉터리에 동봉 → mm_sdk.upload_model 이 Model.zip 에 함께 담음
    #     (rApp/KServe 가 받는 번들에 scaler 가 포함되어 모델과 버전 일치 보장)
    joblib.dump(sc_final, os.path.join(out_dir, 'scaler.pkl'))
    print(f"[scaler] bundled into model dir → {os.path.join(out_dir, 'scaler.pkl')}")
    
    # # (3) 포터블 JSON — sklearn 객체 없이도 추론 측에서 복원 가능
    # scaler_params = {
    #     'type': 'StandardScaler',
    #     'sklearn_version': __import__('sklearn').__version__,
    #     'feature_names': FEATURES,
    #     'n_features_in': int(sc_final.n_features_in_),
    #     'mean': sc_final.mean_.tolist(),
    #     'scale': sc_final.scale_.tolist(),
    #     'var': sc_final.var_.tolist(),
    # }
    # with open(os.path.join(out_dir, 'scaler_params.json'), 'w') as f:
    #     json.dump(scaler_params, f, indent=2)
    # print(f"[scaler] portable params saved → {os.path.join(out_dir, 'scaler_params.json')}")

    #-- 모델 export 및 등록 (원본과 동일)
    model.export(out_dir)                          # TF SavedModel로 export (Keras 3에서 model.save는 .keras/.h5 확장자 필요)
    artifactversion = '1.0.0'                      

    mms_url = ("http://modelmgmtservice.traininghost:8082"          # 사내 모델 관리 서비스
               f"/ai-ml-model-registration/v1/model-registrations/updateArtifact"
               f"/{modelname}/{modelversion}/{artifactversion}")
    print(f"[mms] {requests.post(mms_url).json()}")                # 아티팩트 등록 POST

    mm_sdk = ModelMetricsSdk()                                     # 메트릭 SDK
    trainingjob_id = featurepath.split('_')[-1]                    # 규약: featurepath = <feature_group>_<숫자 trainingjob_id>

    # tm을 거치지 않고 KFP UI에서 직접 실행하면 featurepath에 숫자 jobid 접미사가 없어
    # trainingjob_id가 문자열이 됨 → tm의 update-model-metrics는 정수 id만 받으므로 500.
    # 그 경우 tm 업로드는 건너뛰고 메트릭은 로그로만 남긴다. (추후 InfluxDB→tm 흐름 정착되면 자동 동작)
    if str(trainingjob_id).isdigit():
        mm_sdk.upload_metrics({'metrics': [metrics]}, trainingjob_id)  # 메트릭 업로드
        print(f"[metrics] {mm_sdk.get_metrics(trainingjob_id)}")       # 업로드 결과 조회 확인
    else:
        print(f"[metrics] trainingjob_id='{trainingjob_id}' is not numeric → skip tm upload. metrics={json.dumps(metrics)}")
    mm_sdk.upload_model(out_dir, modelname, modelversion, artifactversion)  # 모델 바이너리 업로드
    print(f"[model] uploaded {modelname}/{modelversion}/{artifactversion}")

# ## 3. 파이프라인 정의 (`@dsl.pipeline`)

#-- 파이프라인 정의
@dsl.pipeline(name="oom_bilstm Pipeline", description="OOM detection — Bi-LSTM")
def super_model_pipeline(
    featurepath: str, modelname: str, modelversion: str,
    hidden_size: int = 128, num_layers: int = 2, window_size: int = 20,
    dropout: float = 0.2, epochs: int = 100, batch_size: int = 256,
    n_splits: int = 5, seed: int = 42,
):
    op = train_export_model(
        featurepath=featurepath, modelname=modelname, modelversion=modelversion,
        hidden_size=hidden_size, num_layers=num_layers, window_size=window_size,
        dropout=dropout, epochs=epochs, batch_size=batch_size,
        n_splits=n_splits, seed=seed,
    )
    op.set_caching_options(False)                           # 캐싱 비활성화 → 매번 다시 실행 (실험 신뢰성↑)
    kubernetes.set_image_pull_policy(op, "IfNotPresent")    # 같은 태그면 재다운로드 안 함 (속도↑)
    op.set_accelerator_type("nvidia.com/gpu")               # GPU 1장 요청 (GPU 이미지 실사용)
    op.set_accelerator_limit(1)
    kubernetes.add_node_selector(op, "gpu", "nvidia")       # GPU 노드(oran-server)에 강제 배치

# ## 4. 파이프라인 컴파일

#-- 컴파일 및 업로드
pipeline_func = super_model_pipeline
file_name = "oom_bilstm_pipeline"
kfp.compiler.Compiler().compile(pipeline_func, f"{file_name}.yaml")  # 파이프라인 → YAML 변환

# ## 5. Training Manager에 업로드
#
# 클러스터 외부(호스트)에서 실행 시: `import os; os.environ['TM_URL']='http://210.123.36.94:32002'` 를 먼저 실행하세요.

import requests                                                       # 이미 위에 있지만 재선언 (중복)
TM_URL = os.environ.get('TM_URL', 'http://tm.traininghost:32002')    # 환경변수 우선, 없으면 기본값
pipeline_name = "oom_bilstm_Pipeline"
resp = requests.post(f"{TM_URL}/pipelines/{pipeline_name}/upload",   # Training Manager로 YAML 업로드
                     files={'file': open(f"{file_name}.yaml", 'rb')})
print(f"[upload] {resp.status_code} {resp.text[:200]}")              # 응답 상태와 본문 일부 출력
