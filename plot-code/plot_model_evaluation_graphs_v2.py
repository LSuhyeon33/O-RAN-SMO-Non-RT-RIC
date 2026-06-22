#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
모델 평가 그래프 생성 코드 v2

수정 반영 사항
1. 라벨/클래스명 변경
   {0: normal, 1: S1-normal, 2: S1-termination, 3: S2, 4: S3, 5: S4}
2. 모든 막대 그래프 위에 수치 표시
3. CPU Usage Comparison은 current_system_usage.csv 파일에서 Max 값만 표시
4. Inference Time Comparison -> Inference Latency Comparison으로 제목 변경
5. Inference latency 비교는 Mean만 표시하며 current_inference_time.csv 파일에서 읽음
6. Memory Usage Comparison은 current_system_usage.csv 파일에서 Max 값만 표시
7. ROC-AUC 커브 코드는 일단 주석 처리
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

LABEL_MAP = {
    0: "normal",
    1: "S1-normal",
    2: "S1-termination",
    3: "S2",
    4: "S3",
    5: "S4",
}

MODEL_COLORS = {
    "XGBoost": "#5354ea",
    "Random Forest": "#dd3bc5",
    "Bi-LSTM": "#ff4c92",
    "LSTM": "#ff8665",
    "TCN": "#ffc352",
    "Transformer": "#f9f871",
}


# =========================
# 2. 데이터 로드 함수
# =========================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_to_name(label) -> str:
    """숫자 라벨을 논문용 클래스명으로 변환한다."""
    try:
        label_int = int(label)
        return LABEL_MAP.get(label_int, str(label))
    except Exception:
        return str(label)


def load_confusion_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    """
    confusion_matrix.csv를 읽는다.

    지원 형식 1:
    actual\predicted,pred_0,pred_1,...
    actual_0,15401,0,...

    지원 형식 2:
    15401,0,0,...
    """
    df = pd.read_csv(path)

    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        raw_labels = df.iloc[:, 0].tolist()
        class_names = []
        for x in raw_labels:
            x = str(x).replace("actual_", "").replace("true_", "").replace("label_", "")
            class_names.append(label_to_name(x))
        matrix = df.iloc[:, 1:].to_numpy(dtype=int)
    else:
        matrix = df.to_numpy(dtype=int)
        class_names = [LABEL_MAP.get(i, str(i)) for i in range(matrix.shape[0])]

    return matrix, class_names


def load_per_class(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


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

    required_cols = ["precision", "recall", "f1"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"per_class.csv에는 '{col}' 열이 필요합니다.")

    df["accuracy"] = compute_class_accuracy_from_cm(cm)
    df["model"] = model_name
    df["class"] = df["label"].apply(label_to_name)

    return df[["model", "class", "accuracy", "precision", "recall", "f1"]]


def build_current_inference_time_df(
    model_name: str,
    metrics: dict,
    latency_samples_path: Path | None = None,
) -> pd.DataFrame:
    """
    현재 모델의 추론 latency 정보를 비교용 형식으로 변환한다.
    current_inference_time.csv가 없을 때 생성용으로 사용한다.
    """
    if latency_samples_path is not None and latency_samples_path.exists():
        lat_df = pd.read_csv(latency_samples_path)

        if "latency_ms" in lat_df.columns:
            latency_col = "latency_ms"
        elif "elapsed_ms" in lat_df.columns:
            latency_col = "elapsed_ms"
        else:
            latency_col = lat_df.select_dtypes(include=[np.number]).columns[0]

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
    current_system_usage.csv가 없을 때 생성용으로 사용한다.
    """
    sys_df = pd.read_csv(system_samples_path)

    if "cpu_percent" not in sys_df.columns:
        numeric_cols = sys_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("system_samples.csv에서 CPU 사용량으로 사용할 숫자형 열을 찾지 못했습니다.")
        cpu_col = numeric_cols[0]
    else:
        cpu_col = "cpu_percent"

    if "rss_mb" in sys_df.columns:
        mem_col = "rss_mb"
    elif "memory_mb" in sys_df.columns:
        mem_col = "memory_mb"
    else:
        numeric_cols = sys_df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in numeric_cols if c != cpu_col]
        if not candidates:
            raise ValueError("system_samples.csv에서 Memory 사용량으로 사용할 숫자형 열을 찾지 못했습니다.")
        mem_col = candidates[0]

    return pd.DataFrame(
        {
            "model": [model_name],
            "cpu_percent_mean": [sys_df[cpu_col].mean()],
            "cpu_percent_max": [sys_df[cpu_col].max()],
            "memory_mb_mean": [sys_df[mem_col].mean()],
            "memory_mb_max": [sys_df[mem_col].max()],
        }
    )


def load_or_create_current_inference_time(
    model_name: str,
    metrics: dict,
    current_inference_time_path: Path,
    latency_samples_path: Path | None,
) -> pd.DataFrame:
    """
    Inference Latency Comparison은 current_inference_time.csv 파일에서 읽어온다.
    파일이 없으면 latency_samples.csv 또는 metrics.json으로 생성 후 저장한다.
    """
    if current_inference_time_path.exists():
        return pd.read_csv(current_inference_time_path)

    df = build_current_inference_time_df(model_name, metrics, latency_samples_path)
    df.to_csv(current_inference_time_path, index=False)
    return df


def load_or_create_current_system_usage(
    model_name: str,
    current_system_usage_path: Path,
    system_samples_path: Path,
) -> pd.DataFrame:
    """
    CPU/Memory Usage Comparison은 current_system_usage.csv 파일에서 읽어온다.
    파일이 없으면 system_samples.csv로 생성 후 저장한다.
    """
    if current_system_usage_path.exists():
        return pd.read_csv(current_system_usage_path)

    df = build_current_system_usage_df(model_name, system_samples_path)
    df.to_csv(current_system_usage_path, index=False)
    return df


def merge_with_optional_compare_csv(current_df: pd.DataFrame, compare_path: Path) -> pd.DataFrame:
    """
    다른 모델 비교용 CSV가 있으면 현재 모델 데이터와 병합한다.
    """
    if not compare_path.exists():
        return current_df

    compare_df = pd.read_csv(compare_path)
    compare_df = compare_df.dropna(how="all")

    if compare_df.empty:
        return current_df

    merged = pd.concat([current_df, compare_df], ignore_index=True)

    if {"model", "class"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["model", "class"], keep="first")
    elif "model" in merged.columns:
        merged = merged.drop_duplicates(subset=["model"], keep="first")

    return merged


# =========================
# 3. 공통 그래프 보조 함수
# =========================

def add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=9, rotation=0):
    """막대 그래프 위에 수치를 표시한다."""
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.01

    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def set_metric_ylim(ax, metric_name: str):
    """성능 지표는 0~1.05 범위로 고정한다."""
    if metric_name in ["accuracy", "precision", "recall", "f1"]:
        ax.set_ylim(0, 1.05)


# =========================
# 4. 그래프 함수
# =========================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    model_name: str,
    normalize: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
        title = f"Normalized Confusion Matrix - {model_name}"
    else:
        plot_cm = cm
        fmt = "d"
        title = f"Confusion Matrix - {model_name}"

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(plot_cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticklabels(class_names)

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
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# ROC-AUC 커브 코드는 일단 주석 처리
# 실제 ROC-AUC Curve를 그리려면 각 샘플의 y_true와 클래스별 예측 확률
# proba_0~proba_5가 포함된 roc_scores.csv가 필요합니다.
# ============================================================
#
# def plot_roc_auc_curve(
#     roc_scores_path: Path,
#     output_path: Path,
#     class_names: list[str],
# ) -> None:
#     try:
#         from sklearn.metrics import roc_curve, auc
#         from sklearn.preprocessing import label_binarize
#     except ImportError as e:
#         raise ImportError(
#             "ROC-AUC 커브를 그리려면 scikit-learn이 필요합니다. pip install scikit-learn"
#         ) from e
#
#     df = pd.read_csv(roc_scores_path)
#     proba_cols = [c for c in df.columns if c.startswith("proba_")]
#     proba_cols = sorted(proba_cols, key=lambda x: int(x.split("_")[1]))
#
#     y_true = df["y_true"].to_numpy()
#     y_score = df[proba_cols].to_numpy()
#
#     classes = list(range(len(proba_cols)))
#     y_true_bin = label_binarize(y_true, classes=classes)
#
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.figure(figsize=(8, 7))
#
#     for i, col in enumerate(proba_cols):
#         fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
#         roc_auc = auc(fpr, tpr)
#         label_name = class_names[i] if i < len(class_names) else str(i)
#         plt.plot(fpr, tpr, linewidth=2, label=f"{label_name} AUC = {roc_auc:.3f}")
#
#     plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, label="Random")
#     plt.title("ROC-AUC Curve", fontsize=15)
#     plt.xlabel("False Positive Rate", fontsize=12)
#     plt.ylabel("True Positive Rate", fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc="lower right", fontsize=8)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()


def plot_roc_auc_summary_bar(metrics: dict, output_path: Path) -> None:
    """
    ROC-AUC 커브 대신 metrics.json의 roc_auc_ovr 값을 요약 막대 그래프로 저장한다.
    """
    roc_auc = metrics.get("roc_auc_ovr", None)
    if roc_auc is None:
        print("[SKIP] roc_auc_ovr 값이 없어 ROC-AUC 요약 그래프를 생성하지 않았습니다.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["OvR ROC-AUC"], [roc_auc], color="tab:blue", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC Summary")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    add_bar_value_labels(ax, bars, fmt="{:.4f}", fontsize=11)

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

    plot_df["class"] = plot_df["class"].apply(label_to_name)

    class_order = list(LABEL_MAP.values())
    model_order = ["XGBoost", "Random Forest", "Bi-LSTM", "LSTM", "TCN", "Transformer"]

    pivot = plot_df.pivot(index="class", columns="model", values=metric)
    pivot = pivot.reindex(index=[c for c in class_order if c in pivot.index])
    pivot = pivot[[m for m in model_order if m in pivot.columns]]

    models = list(pivot.columns)
    classes = list(pivot.index.astype(str))

    x = np.arange(len(classes))
    width = 0.8 / max(len(models), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, model in enumerate(models):
        values = pivot[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, None)
        bars = ax.bar(
            x + idx * width - (len(models) - 1) * width / 2,
            values,
            width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=7, rotation=90)

    ax.set_title(f"Per-class {metric.capitalize()} Comparison", fontsize=15)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    set_metric_ylim(ax, metric)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model", fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_inference_latency_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Inference Latency Comparison
    - current_inference_time.csv에서 읽은 mean_latency_ms만 표시
    """
    required = {"model", "mean_latency_ms"}
    if not required.issubset(df.columns):
        raise ValueError(f"추론 latency 비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df.dropna(subset=["model", "mean_latency_ms"]).copy()
    if plot_df.empty:
        print("[SKIP] inference latency: plot data is empty.")
        return

    model_order = ["XGBoost", "Random Forest", "Bi-LSTM", "LSTM", "TCN", "Transformer"]
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values("model")

    models = plot_df["model"].astype(str).tolist()
    mean_values = plot_df["mean_latency_ms"].to_numpy(dtype=float)
    colors = [MODEL_COLORS.get(m, None) for m in models]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        models,
        mean_values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        label="Mean latency",
    )

    ax.set_title("Inference Latency Comparison", fontsize=15)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # ax.legend()

    add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cpu_usage_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    CPU Usage Comparison
    - current_system_usage.csv에서 cpu_percent_max만 표시
    """
    required = {"model", "cpu_percent_max"}
    if not required.issubset(df.columns):
        raise ValueError(f"CPU 사용량 비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df.dropna(subset=["model", "cpu_percent_max"]).copy()
    if plot_df.empty:
        print("[SKIP] CPU usage: plot data is empty.")
        return

    model_order = ["XGBoost", "Random Forest", "Bi-LSTM", "LSTM", "TCN", "Transformer"]
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values("model")

    models = plot_df["model"].astype(str).tolist()
    values = plot_df["cpu_percent_max"].to_numpy(dtype=float)
    colors = [MODEL_COLORS.get(m, None) for m in models]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        models,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        label="Max CPU",
    )

    ax.set_title("CPU Usage Comparison", fontsize=15)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Max CPU Usage (%)", fontsize=12)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # ax.legend()

    add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_memory_usage_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Memory Usage Comparison
    - current_system_usage.csv에서 memory_mb_max만 표시
    """
    required = {"model", "memory_mb_max"}
    if not required.issubset(df.columns):
        raise ValueError(f"Memory 사용량 비교 데이터에는 {required} 열이 필요합니다.")

    plot_df = df.dropna(subset=["model", "memory_mb_max"]).copy()
    if plot_df.empty:
        print("[SKIP] memory usage: plot data is empty.")
        return

    model_order = ["XGBoost", "Random Forest", "Bi-LSTM", "LSTM", "TCN", "Transformer"]
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values("model")

    models = plot_df["model"].astype(str).tolist()
    values = plot_df["memory_mb_max"].to_numpy(dtype=float)
    colors = [MODEL_COLORS.get(m, None) for m in models]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        models,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        label="Max Memory",
    )

    ax.set_title("Memory Usage Comparison", fontsize=15)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Max Memory Usage (MB)", fontsize=12)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # ax.legend()

    add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =========================
# 5. Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model evaluation graphs v2.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Current model name")
    parser.add_argument("--metrics-json", default="metrics.json")
    parser.add_argument("--confusion-matrix-csv", default="confusion_matrix.csv")
    parser.add_argument("--per-class-csv", default="per_class.csv")
    parser.add_argument("--latency-samples-csv", default="latency_samples.csv")
    parser.add_argument("--system-samples-csv", default="system_samples.csv")
    parser.add_argument("--current-inference-time-csv", default="current_inference_time.csv")
    parser.add_argument("--current-system-usage-csv", default="current_system_usage.csv")
    parser.add_argument("--compare-per-class-csv", default="compare_per_class_metrics.csv")
    parser.add_argument("--compare-inference-time-csv", default="compare_inference_time.csv")
    parser.add_argument("--compare-system-usage-csv", default="compare_system_usage.csv")
    parser.add_argument("--output-dir", default="evaluation_graphs_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_json)
    cm_path = Path(args.confusion_matrix_csv)
    per_class_path = Path(args.per_class_csv)
    latency_path = Path(args.latency_samples_csv)
    system_path = Path(args.system_samples_csv)

    current_inference_time_path = Path(args.current_inference_time_csv)
    current_system_usage_path = Path(args.current_system_usage_csv)

    metrics = load_json(metrics_path)
    cm, class_names = load_confusion_matrix(cm_path)
    per_class_df = load_per_class(per_class_path)

    # 현재 모델 클래스별 성능
    current_per_class_df = build_current_per_class_compare_df(args.model_name, per_class_df, cm)

    # current_inference_time.csv, current_system_usage.csv에서 읽기
    current_inference_df = load_or_create_current_inference_time(
        model_name=args.model_name,
        metrics=metrics,
        current_inference_time_path=current_inference_time_path,
        latency_samples_path=latency_path,
    )

    current_system_df = load_or_create_current_system_usage(
        model_name=args.model_name,
        current_system_usage_path=current_system_usage_path,
        system_samples_path=system_path,
    )

    # # 현재 모델 값 저장
    # current_per_class_df.to_csv(output_dir / "current_per_class_metrics.csv", index=False)
    # current_inference_df.to_csv(output_dir / "current_inference_time.csv", index=False)
    # current_system_df.to_csv(output_dir / "current_system_usage.csv", index=False)

    # # 1. ROC-AUC Curve: 주석 처리
    # plot_roc_auc_curve(
    #     roc_scores_path=Path(args.roc_scores_csv),
    #     output_path=output_dir / "roc_auc_curve.png",
    #     class_names=class_names,
    # )

    # # ROC-AUC 커브 대신 요약 막대 그래프만 생성
    # plot_roc_auc_summary_bar(metrics, output_dir / "roc_auc_summary_bar.png")
    # print(f"Saved: {output_dir / 'roc_auc_summary_bar.png'}")

    # # 2. Confusion Matrix
    # plot_confusion_matrix(
    #     cm,
    #     class_names,
    #     output_dir / "confusion_matrix.png",
    #     model_name=args.model_name,
    #     normalize=False,
    #     )

    # plot_confusion_matrix(
    #     cm,
    #     class_names,
    #     output_dir / "confusion_matrix_normalized.png",
    #     model_name=args.model_name,
    #     normalize=True,
    # )
    # print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    # print(f"Saved: {output_dir / 'confusion_matrix_normalized.png'}")

    # 3. 클래스별 Accuracy, Precision, Recall, F1-score 비교
    per_class_compare_df = merge_with_optional_compare_csv(
        current_per_class_df,
        Path(args.compare_per_class_csv),
    )

    # 비교 CSV에 숫자 class가 있으면 라벨명으로 변환
    if "class" in per_class_compare_df.columns:
        per_class_compare_df["class"] = per_class_compare_df["class"].apply(label_to_name)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_per_class_metric_comparison(
            per_class_compare_df,
            metric,
            output_dir / f"per_class_{metric}_comparison.png",
        )
        print(f"Saved: {output_dir / f'per_class_{metric}_comparison.png'}")

    # 4. Inference Latency Comparison
    inference_compare_df = merge_with_optional_compare_csv(
        current_inference_df,
        Path(args.compare_inference_time_csv),
    )
    plot_inference_latency_comparison(
        inference_compare_df,
        output_dir / "inference_latency_comparison.png",
    )
    print(f"Saved: {output_dir / 'inference_latency_comparison.png'}")

    # 5. CPU, Memory Usage Comparison
    system_compare_df = merge_with_optional_compare_csv(
        current_system_df,
        Path(args.compare_system_usage_csv),
    )

    plot_cpu_usage_comparison(
        system_compare_df,
        output_dir / "cpu_usage_comparison.png",
    )
    print(f"Saved: {output_dir / 'cpu_usage_comparison.png'}")

    plot_memory_usage_comparison(
        system_compare_df,
        output_dir / "memory_usage_comparison.png",
    )
    print(f"Saved: {output_dir / 'memory_usage_comparison.png'}")

    print("\nDone.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
