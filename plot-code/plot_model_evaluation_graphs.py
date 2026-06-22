#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 평가 그래프 생성 코드

생성 가능한 그래프
1. ROC-AUC 커브
2. Confusion Matrix
3. 다른 모델과 클래스별 Accuracy, Precision, Recall, F1-score 비교 막대 그래프
4. 다른 모델과 추론 시간 비교 막대 그래프
5. 다른 모델과 CPU, Memory 사용량 비교 막대 그래프

기본 입력 파일
- metrics.json
- confusion_matrix.csv
- per_class.csv
- latency_samples.csv
- system_samples.csv

선택 입력 파일
- roc_scores.csv
  ROC-AUC 커브를 그리려면 실제 정답과 클래스별 예측 확률이 필요합니다.
  아래 형식으로 준비하세요.

  y_true,proba_0,proba_1,proba_2,proba_3,proba_4,proba_5
  0,0.95,0.01,0.01,0.01,0.01,0.01
  1,0.02,0.80,0.10,0.03,0.03,0.02

비교용 선택 입력 파일
- compare_per_class_metrics.csv
- compare_inference_time.csv
- compare_system_usage.csv

위 비교용 CSV 파일이 없으면 현재 모델 값만 사용해 그래프를 생성합니다.
다른 모델과 비교하려면 해당 CSV에 모델별 값을 추가하면 됩니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

DEFAULT_MODEL_NAME = "Bi-LSTM"
DEFAULT_CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]

# 모델별 색상
MODEL_COLORS = {
    "Bi-LSTM": "tab:blue",
    "LSTM": "tab:orange",
    "TCN": "tab:green",
    "XGBoost": "tab:red",
    "Random Forest": "tab:purple",
    "Transformer": "tab:brown",
}


# =========================
# 2. 데이터 로드 함수
# =========================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    confusion_matrix.csv를 읽는다.

    지원 형식:
    actual\predicted,pred_0,pred_1,...
    actual_0,15401,0,...
    """
    df = pd.read_csv(path)

    # 첫 번째 열이 실제 클래스 이름인 경우 제거
    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        class_names = [str(x).replace("actual_", "") for x in df.iloc[:, 0].tolist()]
        matrix = df.iloc[:, 1:].to_numpy(dtype=int)
    else:
        matrix = df.to_numpy(dtype=int)
        class_names = [str(i) for i in range(matrix.shape[0])]

    return matrix, class_names


def load_per_class(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_class_accuracy_from_cm(cm: np.ndarray) -> list[float]:
    """
    클래스별 accuracy 계산:
    class_i_accuracy = (TP_i + TN_i) / 전체 샘플 수
    """
    total = cm.sum()
    class_accuracy = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        acc = (tp + tn) / total if total > 0 else 0.0
        class_accuracy.append(acc)

    return class_accuracy


def build_current_per_class_compare_df(
    model_name: str,
    per_class_df: pd.DataFrame,
    cm: np.ndarray,
) -> pd.DataFrame:
    """
    현재 모델의 클래스별 accuracy, precision, recall, f1 정보를 비교용 형식으로 변환한다.
    """
    df = per_class_df.copy()

    if "label" not in df.columns:
        raise ValueError("per_class.csv에는 'label' 열이 필요합니다.")

    for col in ["precision", "recall", "f1"]:
        if col not in df.columns:
            raise ValueError(f"per_class.csv에는 '{col}' 열이 필요합니다.")

    df["accuracy"] = compute_class_accuracy_from_cm(cm)
    df["model"] = model_name
    df["class"] = df["label"].astype(str)

    return df[["model", "class", "accuracy", "precision", "recall", "f1"]]


def build_current_inference_time_df(
    model_name: str,
    metrics: dict,
    latency_samples_path: Path | None = None,
) -> pd.DataFrame:
    """
    현재 모델의 추론 시간 정보를 비교용 형식으로 변환한다.
    """
    if latency_samples_path is not None and latency_samples_path.exists():
        lat_df = pd.read_csv(latency_samples_path)
        latency_col = "latency_ms"
        mean_latency = lat_df[latency_col].mean()
        p95_latency = lat_df[latency_col].quantile(0.95)
        p99_latency = lat_df[latency_col].quantile(0.99)
    else:
        latency_info = metrics.get("latency_ms", {})
        mean_latency = latency_info.get("mean", np.nan)
        p95_latency = latency_info.get("p95", np.nan)
        p99_latency = latency_info.get("p99", np.nan)

    return pd.DataFrame(
        {
            "model": [model_name],
            "mean_latency_ms": [mean_latency],
            "p95_latency_ms": [p95_latency],
            "p99_latency_ms": [p99_latency],
        }
    )


def build_current_system_usage_df(
    model_name: str,
    system_samples_path: Path,
) -> pd.DataFrame:
    """
    현재 모델의 CPU, Memory 사용량 정보를 비교용 형식으로 변환한다.
    """
    sys_df = pd.read_csv(system_samples_path)

    if "cpu_percent" not in sys_df.columns:
        raise ValueError("system_samples.csv에는 'cpu_percent' 열이 필요합니다.")

    # 메모리 열 이름 후보
    if "rss_mb" in sys_df.columns:
        mem_col = "rss_mb"
    elif "memory_mb" in sys_df.columns:
        mem_col = "memory_mb"
    else:
        raise ValueError("system_samples.csv에는 'rss_mb' 또는 'memory_mb' 열이 필요합니다.")

    return pd.DataFrame(
        {
            "model": [model_name],
            "cpu_percent_mean": [sys_df["cpu_percent"].mean()],
            "cpu_percent_max": [sys_df["cpu_percent"].max()],
            "memory_mb_mean": [sys_df[mem_col].mean()],
            "memory_mb_max": [sys_df[mem_col].max()],
        }
    )


def merge_with_optional_compare_csv(current_df: pd.DataFrame, compare_path: Path) -> pd.DataFrame:
    """
    비교용 CSV가 있으면 현재 모델 데이터와 병합한다.
    같은 model/class 조합이 있으면 비교용 CSV 값을 우선 사용하지 않고, 현재 모델 값을 유지한다.
    """
    if not compare_path.exists():
        return current_df

    compare_df = pd.read_csv(compare_path)
    compare_df = compare_df.dropna(how="all")

    if compare_df.empty:
        return current_df

    merged = pd.concat([current_df, compare_df], ignore_index=True)

    # 중복 제거 기준
    if {"model", "class"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["model", "class"], keep="first")
    elif "model" in merged.columns:
        merged = merged.drop_duplicates(subset=["model"], keep="first")

    return merged


# =========================
# 3. 그래프 함수
# =========================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = False,
) -> None:
    """
    Confusion Matrix 그래프를 저장한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        plot_cm = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(plot_cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 셀 내부 값 표시
    threshold = plot_cm.max() / 2.0 if plot_cm.max() > 0 else 0
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            value = format(plot_cm[i, j], fmt)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if plot_cm[i, j] > threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_auc_curve(
    roc_scores_path: Path,
    output_path: Path,
    class_names: list[str],
) -> None:
    """
    Multi-class ROC-AUC curve를 저장한다.

    roc_scores.csv 필요 형식:
    - y_true
    - proba_0, proba_1, ..., proba_N

    sklearn이 필요합니다.
    설치:
    pip install scikit-learn
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError as e:
        raise ImportError("ROC-AUC 커브를 그리려면 scikit-learn이 필요합니다. pip install scikit-learn") from e

    if not roc_scores_path.exists():
        raise FileNotFoundError(
            f"ROC-AUC 커브용 파일이 없습니다: {roc_scores_path}\n"
            "y_true와 proba_0~proba_N 열을 포함한 roc_scores.csv를 준비하세요."
        )

    df = pd.read_csv(roc_scores_path)

    if "y_true" not in df.columns:
        raise ValueError("roc_scores.csv에는 'y_true' 열이 필요합니다.")

    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    if not proba_cols:
        raise ValueError("roc_scores.csv에는 'proba_0', 'proba_1' 형태의 확률 열이 필요합니다.")

    proba_cols = sorted(proba_cols, key=lambda x: int(x.split("_")[1]))
    y_true = df["y_true"].to_numpy()
    y_score = df[proba_cols].to_numpy()

    classes = list(range(len(proba_cols)))
    y_true_bin = label_binarize(y_true, classes=classes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))

    # 클래스별 ROC
    for i, col in enumerate(proba_cols):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        label_name = class_names[i] if i < len(class_names) else str(i)
        plt.plot(fpr, tpr, linewidth=2, label=f"Class {label_name} AUC = {roc_auc:.3f}")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", linewidth=2, label=f"Micro-average AUC = {auc_micro:.3f}")

    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, label="Random")
    plt.title("ROC-AUC Curve", fontsize=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_auc_summary_bar(metrics: dict, output_path: Path) -> None:
    """
    실제 ROC curve 입력이 없을 때, metrics.json의 roc_auc_ovr 값을 요약 막대 그래프로 저장한다.
    """
    roc_auc = metrics.get("roc_auc_ovr", None)
    if roc_auc is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.bar(["OvR ROC-AUC"], [roc_auc], color="tab:blue", edgecolor="black")
    plt.ylim(0, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC Summary")
    plt.text(0, roc_auc + 0.02, f"{roc_auc:.4f}", ha="center", va="bottom", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """
    클래스별 metric 비교 막대 그래프를 저장한다.
    metric: accuracy, precision, recall, f1
    """
    required = {"model", "class", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df[["model", "class", metric]].dropna()
    if plot_df.empty:
        print(f"[SKIP] {metric}: plot data is empty.")
        return

    pivot = plot_df.pivot(index="class", columns="model", values=metric)
    models = list(pivot.columns)
    classes = list(pivot.index.astype(str))

    x = np.arange(len(classes))
    width = 0.8 / max(len(models), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))

    for idx, model in enumerate(models):
        values = pivot[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, None)
        plt.bar(
            x + idx * width - (len(models) - 1) * width / 2,
            values,
            width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    plt.title(f"Per-class {metric.capitalize()} Comparison", fontsize=15)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xticks(x, classes)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Model", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_inference_time_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    모델별 추론 시간 비교 막대 그래프를 저장한다.
    mean latency와 p95 latency를 함께 표시한다.
    """
    required = {"model", "mean_latency_ms"}
    if not required.issubset(df.columns):
        raise ValueError(f"추론 시간 비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df.dropna(subset=["model", "mean_latency_ms"]).copy()
    if plot_df.empty:
        print("[SKIP] inference time: plot data is empty.")
        return

    models = plot_df["model"].astype(str).tolist()
    mean_values = plot_df["mean_latency_ms"].to_numpy(dtype=float)

    has_p95 = "p95_latency_ms" in plot_df.columns and plot_df["p95_latency_ms"].notna().any()
    p95_values = plot_df["p95_latency_ms"].to_numpy(dtype=float) if has_p95 else None

    x = np.arange(len(models))
    width = 0.35 if has_p95 else 0.6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2 if has_p95 else x,
        mean_values,
        width,
        label="Mean latency",
        edgecolor="black",
        linewidth=0.6,
    )

    if has_p95:
        plt.bar(
            x + width / 2,
            p95_values,
            width,
            label="P95 latency",
            edgecolor="black",
            linewidth=0.6,
        )

    plt.title("Inference Time Comparison", fontsize=15)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()

    # 평균값 수치 표시
    for i, value in enumerate(mean_values):
        xpos = x[i] - width / 2 if has_p95 else x[i]
        plt.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    if has_p95:
        for i, value in enumerate(p95_values):
            plt.text(x[i] + width / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_system_usage_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    CPU, Memory 사용량 비교 막대 그래프를 저장한다.
    서로 단위가 다르므로 CPU와 Memory를 별도 이미지로 저장한다.
    """
    required = {"model", "cpu_percent_mean", "memory_mb_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"시스템 사용량 비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df.dropna(subset=["model"]).copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    # CPU mean/max
    cpu_cols = [c for c in ["cpu_percent_mean", "cpu_percent_max"] if c in plot_df.columns]
    cpu_df = plot_df.dropna(subset=["cpu_percent_mean"])
    if not cpu_df.empty:
        x = np.arange(len(cpu_df))
        width = 0.35 if "cpu_percent_max" in cpu_cols else 0.6

        plt.figure(figsize=(10, 6))
        plt.bar(
            x - width / 2 if "cpu_percent_max" in cpu_cols else x,
            cpu_df["cpu_percent_mean"],
            width,
            label="Mean CPU",
            edgecolor="black",
            linewidth=0.6,
        )
        if "cpu_percent_max" in cpu_cols:
            plt.bar(
                x + width / 2,
                cpu_df["cpu_percent_max"],
                width,
                label="Max CPU",
                edgecolor="black",
                linewidth=0.6,
            )

        plt.title("CPU Usage Comparison", fontsize=15)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("CPU Usage (%)", fontsize=12)
        plt.xticks(x, cpu_df["model"].astype(str), rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cpu_usage_comparison.png", dpi=300)
        plt.close()

    # Memory mean/max
    mem_cols = [c for c in ["memory_mb_mean", "memory_mb_max"] if c in plot_df.columns]
    mem_df = plot_df.dropna(subset=["memory_mb_mean"])
    if not mem_df.empty:
        x = np.arange(len(mem_df))
        width = 0.35 if "memory_mb_max" in mem_cols else 0.6

        plt.figure(figsize=(10, 6))
        plt.bar(
            x - width / 2 if "memory_mb_max" in mem_cols else x,
            mem_df["memory_mb_mean"],
            width,
            label="Mean Memory",
            edgecolor="black",
            linewidth=0.6,
        )
        if "memory_mb_max" in mem_cols:
            plt.bar(
                x + width / 2,
                mem_df["memory_mb_max"],
                width,
                label="Max Memory",
                edgecolor="black",
                linewidth=0.6,
            )

        plt.title("Memory Usage Comparison", fontsize=15)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Memory Usage (MB)", fontsize=12)
        plt.xticks(x, mem_df["model"].astype(str), rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "memory_usage_comparison.png", dpi=300)
        plt.close()


# =========================
# 4. 비교용 CSV 템플릿 생성
# =========================

def create_template_csvs(
    output_dir: Path,
    current_per_class_df: pd.DataFrame,
    current_inference_df: pd.DataFrame,
    current_system_df: pd.DataFrame,
) -> None:
    """
    다른 모델 값을 입력할 수 있는 비교용 CSV 템플릿을 생성한다.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_list = ["LSTM", "TCN", "XGBoost", "Random Forest", "Transformer"]
    class_list = sorted(current_per_class_df["class"].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)

    # 클래스별 성능 비교 템플릿
    rows = []
    for model in model_list:
        for cls in class_list:
            rows.append(
                {
                    "model": model,
                    "class": cls,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "compare_per_class_metrics_template.csv", index=False)

    # 추론 시간 비교 템플릿
    pd.DataFrame(
        {
            "model": model_list,
            "mean_latency_ms": [np.nan] * len(model_list),
            "p95_latency_ms": [np.nan] * len(model_list),
            "p99_latency_ms": [np.nan] * len(model_list),
        }
    ).to_csv(output_dir / "compare_inference_time_template.csv", index=False)

    # CPU/Memory 비교 템플릿
    pd.DataFrame(
        {
            "model": model_list,
            "cpu_percent_mean": [np.nan] * len(model_list),
            "cpu_percent_max": [np.nan] * len(model_list),
            "memory_mb_mean": [np.nan] * len(model_list),
            "memory_mb_max": [np.nan] * len(model_list),
        }
    ).to_csv(output_dir / "compare_system_usage_template.csv", index=False)


# =========================
# 5. Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model evaluation graphs.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Current model name")
    parser.add_argument("--metrics-json", default="metrics.json")
    parser.add_argument("--confusion-matrix-csv", default="confusion_matrix.csv")
    parser.add_argument("--per-class-csv", default="per_class.csv")
    parser.add_argument("--latency-samples-csv", default="latency_samples.csv")
    parser.add_argument("--system-samples-csv", default="system_samples.csv")
    parser.add_argument("--roc-scores-csv", default="roc_scores.csv")
    parser.add_argument("--compare-per-class-csv", default="compare_per_class_metrics.csv")
    parser.add_argument("--compare-inference-time-csv", default="compare_inference_time.csv")
    parser.add_argument("--compare-system-usage-csv", default="compare_system_usage.csv")
    parser.add_argument("--output-dir", default="evaluation_graphs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_json)
    cm_path = Path(args.confusion_matrix_csv)
    per_class_path = Path(args.per_class_csv)
    latency_path = Path(args.latency_samples_csv)
    system_path = Path(args.system_samples_csv)
    roc_scores_path = Path(args.roc_scores_csv)

    metrics = load_json(metrics_path)
    cm, class_names = load_confusion_matrix(cm_path)
    per_class_df = load_per_class(per_class_path)

    current_per_class_df = build_current_per_class_compare_df(args.model_name, per_class_df, cm)
    current_inference_df = build_current_inference_time_df(args.model_name, metrics, latency_path)
    current_system_df = build_current_system_usage_df(args.model_name, system_path)

    # 템플릿 CSV 생성
    create_template_csvs(output_dir, current_per_class_df, current_inference_df, current_system_df)

    # 1. ROC-AUC Curve
    try:
        plot_roc_auc_curve(
            roc_scores_path=roc_scores_path,
            output_path=output_dir / "roc_auc_curve.png",
            class_names=class_names,
        )
        print(f"Saved: {output_dir / 'roc_auc_curve.png'}")
    except Exception as e:
        print(f"[WARNING] ROC-AUC curve was not created: {e}")
        print("[INFO] metrics.json의 roc_auc_ovr 값으로 요약 그래프를 생성합니다.")
        plot_roc_auc_summary_bar(metrics, output_dir / "roc_auc_summary_bar.png")
        print(f"Saved: {output_dir / 'roc_auc_summary_bar.png'}")

    # 2. Confusion Matrix
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix_normalized.png", normalize=True)
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved: {output_dir / 'confusion_matrix_normalized.png'}")

    # 3. 클래스별 Accuracy, Precision, Recall, F1-score 비교
    per_class_compare_df = merge_with_optional_compare_csv(
        current_per_class_df,
        Path(args.compare_per_class_csv),
    )
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_per_class_metric_comparison(
            per_class_compare_df,
            metric,
            output_dir / f"per_class_{metric}_comparison.png",
        )
        print(f"Saved: {output_dir / f'per_class_{metric}_comparison.png'}")

    # 4. 추론 시간 비교
    inference_compare_df = merge_with_optional_compare_csv(
        current_inference_df,
        Path(args.compare_inference_time_csv),
    )
    plot_inference_time_comparison(
        inference_compare_df,
        output_dir / "inference_time_comparison.png",
    )
    print(f"Saved: {output_dir / 'inference_time_comparison.png'}")

    # 5. CPU, Memory 사용량 비교
    system_compare_df = merge_with_optional_compare_csv(
        current_system_df,
        Path(args.compare_system_usage_csv),
    )
    plot_system_usage_comparison(system_compare_df, output_dir)
    print(f"Saved: {output_dir / 'cpu_usage_comparison.png'}")
    print(f"Saved: {output_dir / 'memory_usage_comparison.png'}")

    # 현재 모델 값도 CSV로 저장
    current_per_class_df.to_csv(output_dir / "current_per_class_metrics.csv", index=False)
    current_inference_df.to_csv(output_dir / "current_inference_time.csv", index=False)
    current_system_df.to_csv(output_dir / "current_system_usage.csv", index=False)

    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
