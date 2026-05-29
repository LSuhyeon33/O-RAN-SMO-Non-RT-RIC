# Task: OOM 파이프라인 6종 5-fold CV + 메트릭 스키마 통일

계획서: /root/.claude/plans/4-bilstm-misty-cascade.md

## 체크리스트
- [x] 1. bilstm — .ipynb + .py 메트릭 보강(FNR/AUC/CM/Latency) + Output[Artifact] 3종 + py_compile ✅
- [x] 2. lstm — .ipynb + .py 5-fold CV 전환 + Output[Artifact] + py_compile ✅
- [x] 3. tcn — .ipynb + .py 5-fold CV 전환 + Output[Artifact] + py_compile ✅
- [x] 4. transformer — .ipynb + .py 5-fold CV 전환 + Output[Artifact] + py_compile ✅
- [x] 5. rf — .ipynb + .py 5-fold CV 전환 + Output[Artifact] + py_compile ✅
- [x] 6. xgboost — .ipynb + .py 5-fold CV 전환 + Output[Artifact] + py_compile ✅
- [x] 7. 6개 .py 일괄 py_compile + metrics dict 키 일치 리뷰 ✅

## Review (완료)

### 변경 파일 (12개)
icsr-pipelines/oom_pipeline_{bilstm,lstm,tcn,transformer,rf,xgboost}.{ipynb,py}

### 적용 내용
- **lstm/tcn/transformer/rf/xgboost**: 단일 hold-out(`val_per_class`) → 5-fold
  `StratifiedKFold` 교차검증. CV 후 전체 데이터로 최종 모델 재학습.
  - DL 3종: 윈도우 먼저 생성 → 윈도우 단위 fold 분할 → fold별 Scaler/모델/EarlyStopping
    → CV 평균 best_epoch로 전체 재학습 (bilstm 패턴).
  - rf/xgboost: 행 단위 fold 분할 → fold별 SMOTE→Scaler→fit (leakage 방지) →
    전체 SMOTE+Scaler로 최종 재학습.
- **bilstm**: CV 구조 유지, 메트릭만 보강 — FNR/AUC/F1_weighted/ConfusionMatrix를
  fold별 계산, 최종 모델 Latency 측정 추가.
- **6종 공통**: 통일 메트릭 dict (DL 18키 / RF·XGB 17키 = AvgBestEpoch 제외).
  모든 메트릭 fold 평균, ConfusionMatrix는 fold 합산(전체 OOF).
- **6종 공통**: `cv_results.json` + `training_history.csv/json`을 KFP `Output[Artifact]`
  3종으로 추출 (기존 out_dir 동봉도 유지). `val_per_class` → `n_splits=5`.
- 부수 수정: lstm/tcn/transformer/rf/xgboost `.ipynb`의 파이프라인 정의가
  두 셀로 쪼개져 셀 단독 실행 시 SyntaxError 나던 것 → bilstm처럼 온전한 셀로 정리.

### 검증 결과
- `python3 -m py_compile` 6개 .py 전부 통과.
- 6개 .ipynb ↔ .py 코드 본문(imports/component/pipeline/compile) 일치 확인.
- 6개 .ipynb 전 코드 셀(각 5개) 문법 검증 통과.
- metrics dict 키 일관성 확인 (DL 4종 동일 18키, RF/XGB 동일 17키).
- ⚠ kfp 로컬 미설치 → `Compiler().compile()` 실검증은 못 함 (문법+로직 리뷰로 갈음).

### 미수정 (사용자 범위 제외 — 후속 필요)
- `README.md`: "데이터 분할 정책 / 평가 메트릭 / 파라미터 매트릭스" 절이 변경 내용과
  불일치 (val_per_class·단일 hold-out 기준 서술).
- `_common_reference.py`: `split_validation` 헬퍼가 이제 어느 파이프라인에서도
  쓰이지 않음 (CV 헬퍼 미추가).
