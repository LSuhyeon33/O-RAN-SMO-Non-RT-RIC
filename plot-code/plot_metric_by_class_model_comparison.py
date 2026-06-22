#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
클래스별 모델 탐지 성능 비교 막대 그래프 생성 코드

그래프 형식:
- 가로축: 클래스
- 세로축: 평가 지표
  - accuracy
  - precision
  - recall
  - f1
- 색상: 모델 구분
- 범례: 모델명

입력 CSV 예시:
model,class,accuracy,precision,recall,f1
XGBoost,0,1.0,1.0,1.0,1.0
XGBoost,1,0.034,0.069,0.034,0.046
Bi-LSTM,0,0.99,0.99,0.99,0.99

사용 예시:
1) 특정 지표 하나만 출력
python plot_metric_by_class_model_comparison.py \
  --input current_per_class_metrics.csv \
  --metric f1 \
  --output-dir metric_by_class_model_graphs

2) accuracy, precision, recall, f1 전체 출력
python plot_metric_by_class_model_comparison.py \
  --input current_per_class_metrics.csv \
  --metric all \
  --output-dir metric_by_class_model_graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABEL_MAP = {
    0: "normal",
    1: "S1-normal",
    2: "S1-termination",
    3: "S2",
    4: "S3",
    5: "S4",
}

CLASS_ORDER = [
    "normal",
    "S1-normal",
    "S1-termination",
    "S2",
    "S3",
    "S4",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}

MODEL_COLORS = {
    "XGBoost": "#5354ea",
    "Random Forest": "#dd3bc5",
    "Bi-LSTM": "#ff4c92",
    "LSTM": "#ff8665",
    "TCN": "#ffc352",
    "Transformer": "#f9f871",
}



def label_to_name(label) -> str:
    """숫자 클래스 라벨을 클래스명으로 변환한다."""
    try:
        label_int = int(label)
        return LABEL_MAP.get(label_int, str(label))
    except Exception:
        return str(label)


def load_metrics_csv(input_path: Path) -> pd.DataFrame:
    """CSV를 읽고 클래스명 변환 및 필수 컬럼 검증을 수행한다."""
    df = pd.read_csv(input_path)

    required_cols = {"model", "class", "accuracy", "precision", "recall", "f1"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"CSV 파일에 필요한 컬럼이 없습니다: {missing_cols}")

    df["class_name"] = df["class"].apply(label_to_name)

    df["class_name"] = pd.Categorical(
        df["class_name"],
        categories=CLASS_ORDER,
        ordered=True,
    )

    df = df.sort_values(["class_name", "model"])

    return df


def add_bar_value_labels(ax, bars, fmt="{:.2f}", fontsize=8, rotation=90):
    """막대 위에 수치 표시."""
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


def plot_metric_by_class_model(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
):
    """
    x축: class
    y축: metric
    색상: model
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"지원하지 않는 metric입니다: {metric}")

    plot_df = df[["model", "class_name", metric]].dropna()

    if plot_df.empty:
        raise ValueError(f"{metric} 그래프를 그릴 데이터가 없습니다.")

    # class x model pivot
    pivot = plot_df.pivot_table(
        index="class_name",
        columns="model",
        values=metric,
        aggfunc="mean",
        observed=False,
    )

    # 클래스 순서 고정
    pivot = pivot.reindex(CLASS_ORDER)
    pivot = pivot.dropna(how="all")

    # 모델 순서 고정 + CSV에 있는 모델만 사용
    model_order = [
        "XGBoost",
        "Random Forest",
        "Bi-LSTM",
        "LSTM",
        "TCN",
        "Transformer",
    ]
    models = [m for m in model_order if m in pivot.columns]

    # 그 외 모델도 뒤에 추가
    extra_models = [m for m in pivot.columns if m not in models]
    models.extend(extra_models)

    pivot = pivot[models]

    classes = pivot.index.astype(str).tolist()
    x = np.arange(len(classes))

    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, model in enumerate(models):
        values = pivot[model].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, None)

        bars = ax.bar(
            x + idx * width - (n_models - 1) * width / 2,
            values,
            width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

        add_bar_value_labels(
            ax,
            bars,
            fmt="{:.2f}",
            fontsize=8 if n_models <= 3 else 7,
            rotation=90 if n_models >= 3 else 0,
        )

    ax.set_title(f"Per-class {METRIC_LABELS[metric]} Comparison", fontsize=15)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")

    ax.set_ylim(0.0, 1.10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.legend(
        title="Model",
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-class metric comparison grouped by model."
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument(
        "--metric",
        default="all",
        choices=["accuracy", "precision", "recall", "f1", "all"],
        help="Metric to plot",
    )
    parser.add_argument(
        "--output-dir",
        default="metric_by_class_model_graphs",
        help="Output directory",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics_csv(input_path)

    if args.metric == "all":
        metrics = ["accuracy", "precision", "recall", "f1"]
    else:
        metrics = [args.metric]

    for metric in metrics:
        plot_metric_by_class_model(
            df=df,
            metric=metric,
            output_path=output_dir / f"per_class_{metric}_by_model.png",
        )


if __name__ == "__main__":
    main()
