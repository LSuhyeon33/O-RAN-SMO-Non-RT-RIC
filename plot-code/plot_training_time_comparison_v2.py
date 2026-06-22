#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6개 모델의 학습 시간을 비교하는 막대 그래프 생성 코드

사용 방법:
1. 아래 MODEL_NAMES와 TRAINING_TIMES 값을 직접 수정한다.
2. TRAINING_TIMES에는 각 모델의 학습 시간을 숫자로 입력한다.
   예: 초 단위라면 120.5, 분 단위라면 2.01처럼 입력
3. 실행하면 training_time_comparison.png 파일이 생성된다.

특징:
- 6개 모델별 막대 색상을 서로 다르게 표시
- legend에 각 모델명을 표시
- 막대 위에 학습 시간 수치 표시
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# 1. 사용자 입력 영역
# =========================

MODEL_NAMES = [
    "XGBoost",
    "Random Forest",
    "Bi-LSTM",
    "LSTM",
    "TCN",
    "Transformer",
]

# 학습 시간을 직접 입력하세요.
# 예시:
# TRAINING_TIMES = [120.5, 98.2, 75.3, 12.8, 8.4, 150.6]
TRAINING_TIMES = [
    3.68,  # XGBoost
    2.23,  # Random Forest
    19.9,  # Bi-LSTM
    19.52,  # LSTM
    50.38,  # TCN
    38.63,  # Transformer
]

# 모델별 막대 색상
BAR_COLORS = [
    "#5354ea",
    "#dd3bc5",
    "#ff4c92",
    "#ff8665",
    "#ffc352",
    "#f9f871",
]

# y축 단위 이름을 원하는 대로 수정하세요.
# 예: "Training Time (s)", "Training Time (min)", "Training Time (hour)"
Y_LABEL = "Training Time (min)"

# 그래프 제목
TITLE = "Training Time Comparison"

# 출력 이미지 파일명
OUTPUT_PATH = "training_time_comparison.png"


# =========================
# 2. 그래프 생성 함수
# =========================

def validate_inputs(model_names, training_times, bar_colors):
    """입력값을 검증한다."""
    if len(model_names) != len(training_times):
        raise ValueError(
            f"MODEL_NAMES 길이({len(model_names)})와 "
            f"TRAINING_TIMES 길이({len(training_times)})가 다릅니다."
        )

    if len(model_names) != len(bar_colors):
        raise ValueError(
            f"MODEL_NAMES 길이({len(model_names)})와 "
            f"BAR_COLORS 길이({len(bar_colors)})가 다릅니다."
        )

    if any(t is None for t in training_times):
        empty_indices = [i for i, t in enumerate(training_times) if t is None]
        empty_models = [model_names[i] for i in empty_indices]
        raise ValueError(
            "TRAINING_TIMES에 아직 입력되지 않은 값이 있습니다. "
            f"다음 모델의 학습 시간을 입력하세요: {empty_models}"
        )

    if any(not isinstance(t, (int, float)) for t in training_times):
        raise TypeError("TRAINING_TIMES에는 숫자형 값만 입력해야 합니다.")

    if any(t < 0 for t in training_times):
        raise ValueError("학습 시간은 음수가 될 수 없습니다.")


def plot_training_time_bar_chart(
    model_names,
    training_times,
    bar_colors,
    y_label="Training Time (min)",
    title="Training Time Comparison",
    output_path="training_time_comparison.png",
):
    """6개 모델의 학습 시간을 색상이 다른 막대 그래프로 저장한다."""
    validate_inputs(model_names, training_times, bar_colors)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        model_names,
        training_times,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Model", fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # 막대 위에 수치 표시
    max_time = max(training_times)
    offset = max_time * 0.01 if max_time > 0 else 0.01

    for bar, value in zip(bars, training_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Legend 추가
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=model)
        for model, color in zip(model_names, bar_colors)
    ]
    plt.legend(
        handles=legend_handles,
        title="Model",
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
    )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph to: {output_path}")


# =========================
# 3. 실행 영역
# =========================

if __name__ == "__main__":
    plot_training_time_bar_chart(
        model_names=MODEL_NAMES,
        training_times=TRAINING_TIMES,
        bar_colors=BAR_COLORS,
        y_label=Y_LABEL,
        title=TITLE,
        output_path=OUTPUT_PATH,
    )
