# OOM Detection Pipelines (IDS rApp 실험 1 — C1)

`IDS_rApp_Chapter4_실험설계.docx` 4.3절에 따라 O-RAN OOM 공격 탐지를 위한 6개 모델 KFP(Kubeflow Pipelines) 학습 파이프라인 구현.

## 산출물

```
pipelines/
├── _common_reference.py         # 공통 헬퍼 참고용 (import 안 됨, 복붙 원본)
├── oom_pipeline_xgboost.py      # ML — XGBoost
├── oom_pipeline_rf.py           # ML — Random Forest
├── oom_pipeline_lstm.py         # DL — LSTM (sliding window)
├── oom_pipeline_bilstm.py       # DL — Bidirectional LSTM
├── oom_pipeline_tcn.py          # DL — Temporal Convolutional Network
├── oom_pipeline_transformer.py  # DL — Encoder-only Transformer
├── oom_pipeline_*.ipynb         # 위 6개 .py의 노트북 버전 (aiml-notebook 환경에서 셀 단위 실행용 — 코드 내용은 .py와 동일)
└── README.md
```

## 입력 데이터

| 항목 | 값 |
|---|---|
| 위치 | Cassandra `feature_store.oom_b_riclog` |
| 레코드 수 | 514,960 |
| 입력 피처 | 20개 집계 피처 (E2AP/RIC log 파생) |
| 레이블 | `label` (6-class: 0=baseline, 1~5=공격 시나리오) |
| 정렬 | **Cassandra 적재 시점에 ts_ns 정렬 완료** (파이프라인은 전체 데이터로 5-fold CV 수행) |

## 검증 정책 (5-fold 교차검증)

6개 모델 모두 **Stratified 5-fold 교차검증**으로 평가한다
(`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

- **DL (LSTM/Bi-LSTM/TCN/Transformer)**: 전체 데이터를 슬라이딩 윈도우로 만든 뒤
  윈도우 단위로 fold 분할. fold마다 `StandardScaler`·모델·`EarlyStopping`을 새로 시작.
- **ML (XGBoost/RF)**: 원본 행 단위로 fold 분할. fold마다 SMOTE→`StandardScaler`→fit을
  학습셋 내부에서만 수행 (data leakage 방지).
- **최종 모델**: CV 후 **전체 데이터로 재학습**해 export·등록한다.
  DL은 CV 평균 best epoch만큼, ML은 그대로 재학습.
- **Test**: **본 파이프라인에서 수행 안 함** (rApp 추론 단계에서 별도 수행).

## SMOTE 처리

| 모델 종류 | SMOTE | 대안 |
|---|---|---|
| XGBoost / RF (윈도우 없음) | ✅ fold별 학습셋에만 적용 (`random_state=42`, leakage 방지) | — |
| LSTM / Bi-LSTM / TCN / Transformer (sliding window) | ❌ 시퀀스 시간 구조 보존 | `sklearn.utils.class_weight.compute_class_weight('balanced')`로 산출한 `class_weight`를 `model.fit`에 전달 |

(docx S4와의 의식적 편차 — 시퀀스 모델에 SMOTE를 적용하면 윈도우 내 시간 의존성이 깨지므로 표준 우회법 채택)

## 평가 메트릭 (5-fold CV 평균)

모든 메트릭을 fold별로 계산한 뒤 평균한다. `ConfusionMatrix`만 fold 합산
(StratifiedKFold가 표본을 분할하므로 합 = 전체 OOF 행렬). 업로드 `metrics` dict는
6개 모델 공통 스키마:

```
AvgValLoss / StdValLoss / FoldValLosses   fold 검증 손실 (DL=val_loss, ML=log_loss)
AvgAccuracy                                fold 평균 정확도
AvgF1Macro / AvgF1Weighted                 precision_recall_fscore_support
AvgPrecisionMacro / AvgRecallMacro         macro 평균
AvgFNR / AvgFNRPerClass                    FN_i/(FN_i+TP_i) — 보안 관점 핵심 지표
AvgROCAUCovr                               roc_auc_score(multi_class='ovr')
ConfusionMatrix                            6×6 정수 행렬 (fold 합산 = 전체 OOF)
Latency_ms_mean/std                        최종 모델, 배치 1건 warmup 100 / 측정 1000회
AvgBestEpoch                               DL 전용 (CV가 찾은 평균 best epoch)
NSplits / FoldDetails / Hyperparameters    fold 수 / fold별 상세 / 하이퍼파라미터
```

DL 4종은 18키, ML 2종(RF/XGBoost)은 `AvgBestEpoch`을 뺀 17키다.

## 산출 파일

각 파이프라인은 모델 디렉터리(`./`)에 아래를 저장하고 `mm_sdk.upload_model("./", ...)`로
모델과 함께 LeoFS/MinIO·모델 등록부에 업로드한다 (= `Model.zip`에 포함).

- `cv_results.json` — fold별 메트릭 + CV 평균 + 하이퍼파라미터
- `training_history.csv` / `training_history.json` — 학습 곡선

| 모델 | training_history 캡처 방식 |
|---|---|
| LSTM / Bi-LSTM / TCN / Transformer | 최종 재학습의 `CSVLogger` + `history.history` (epoch별) |
| XGBoost | 최종 재학습 `eval_set` → `evals_result_` (iteration별 train mlogloss) |
| Random Forest | epoch 개념 없음 → fold별 CV 결과 행 (`fold`/`val_logloss`/`val_accuracy`) |

`cv_results.json`·`training_history.csv`·`.json`은 KFP `Output[Artifact]`로도 노출되어
KFP UI에서 직접 다운로드 가능. **Bi-LSTM**은 추가로 최종 `StandardScaler`를 모델
디렉터리에 `scaler.pkl`로 동봉(+ KFP 아티팩트로도 추출) → rApp이 모델 번들에서
같은 scaler를 받아 쓴다.

## 파이프라인 파라미터 매트릭스

| 파일 | 파라미터 | Default |
|---|---|---|
| **공통** | `featurepath`, `modelname`, `modelversion`, `n_splits`, `seed` | —, —, —, 5, 42 |
| `oom_pipeline_xgboost.py` | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` | 300, 6, 0.1, 1.0 |
| `oom_pipeline_rf.py` | `n_estimators`, `max_depth` (str: 'None'/'10'/'20') | 300, 'None' |
| `oom_pipeline_lstm.py` | `hidden_size`, `num_layers`, `window_size`, `dropout`, `epochs`, `batch_size` | 128, 2, 20, 0.2, 50, 256 |
| `oom_pipeline_bilstm.py` | LSTM과 동일 (단 `epochs` 기본값 100) | 128, 2, 20, 0.2, 100, 256 |
| `oom_pipeline_tcn.py` | `nb_filters`, `kernel_size`, `dilations_str`, `nb_stacks`, `dropout`, `window_size`, `epochs`, `batch_size` | 64, 3, '1,2,4,8', 1, 0.1, 20, 50, 256 |
| `oom_pipeline_transformer.py` | `d_model`, `nhead`, `num_layers`, `ff_dim`, `window_size`, `dropout`, `epochs`, `batch_size` | 128, 8, 2, 256, 20, 0.1, 50, 256 |

HP 탐색은 외부에서 (tm 학습 조건 변경 또는 별도 sweep 스크립트) 진행. 본 파일들은 단일 HP 조합 실행만 담당.

## 컴파일 + 업로드

각 파이프라인 파일은 실행 시 (a) `*.yaml` 컴파일, (b) tm으로 POST 업로드까지 자동 수행한다.

### 호스트(`oran-server`)에서 실행

```bash
cd /usr/local/o-ran/pipelines

# tm은 cluster-internal DNS(tm.traininghost)이므로 호스트 외부에서는 NodePort로 우회
export TM_URL=http://210.123.36.94:32002

# 한 번에 모두 컴파일·업로드
for f in oom_pipeline_*.py; do
  echo "=== $f ==="
  python3 "$f" || break
done
```

### 클러스터 내부(파드 안)에서 실행

```bash
# 환경변수 미설정 → 기본 cluster DNS 사용
unset TM_URL
python3 oom_pipeline_xgboost.py
```

### 확인

```bash
ls /usr/local/o-ran/pipelines/*.yaml | wc -l   # → 6
curl -s http://210.123.36.94:32002/pipelines/ | jq '.[].pipeline_name'
```

## 학습 실행 (예: tm REST 또는 UI에서)

학습 작업 생성 시 파라미터 예:
```json
{
  "modelname": "oom-xgb",
  "modelversion": "1.0.0",
  "featurepath": "oom_b_riclog_<trainingjob_id>",
  "n_estimators": 300,
  "max_depth": 6
}
```

`featurepath`는 `<feature_group>_<trainingjob_id>` 형식. tm이 학습 작업 생성 시 자동 채움.

## 학습 결과 회수

- **메트릭**: `mm_sdk.get_metrics(trainingjob_id)` 또는 `modelmgmtservice` REST
- **모델 + 산출물**: 모델 등록부(`http://210.123.36.94:32006`) → 모델 다운로드 → 압축 해제하면 `cv_results.json` · `training_history.csv/json` 동봉 (Bi-LSTM은 `scaler.pkl`도 포함)

## 후속 작업 (이 디렉터리 범위 외)

- 탐지 rApp 1개 + baseline rApp 1개 구현 (실험 2)
- HP grid sweep 자동화 (CV 평균 `AvgValLoss` 기준으로 HP 조합 선택)
- 결과 비교 스크립트 (docx Table 8 채우기)
