"""
6개 모델 성능 비교 꺾은선 그래프
- 가로축: 평가지표 (Accuracy, F1-Score, Precision, Recall, FNR)
- 세로축: 성능 수치 (%)
- 꺾은선: 모델
"""
import matplotlib.pyplot as plt

# ── 데이터 (업로드한 표와 동일) ──────────────────────────────
metrics = ["Accuracy", "F1-Score", "Precision", "Recall", "FNR"]

data = {
    # 모델명: [Accuracy, F1-Score, Precision, Recall, FNR]
    "Bi-LSTM":       [86.225, 85.116, 88.711, 86.225, 13.775],
    "LSTM":          [72.367, 66.182, 72.075, 72.367, 27.633],
    "TCN":           [73.592, 68.170, 69.933, 73.592, 26.408],
    "Transformer":   [70.045, 66.212, 69.911, 70.045, 29.955],
    "XGBoost":       [60.584, 57.295, 63.991, 60.584, 39.416],
    "Random Forest": [58.113, 53.637, 61.401, 58.113, 41.887],
}

# 각 모델마다 색/마커를 구분
markers = ["o", "s", "^", "D", "v", "P"]

# ── 그래프 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

for (model, values), marker in zip(data.items(), markers):
    ax.plot(metrics, values, marker=marker, markersize=8,
            linewidth=2, label=model)
    # 각 점에 수치 라벨 표시 (원하지 않으면 이 블록 제거)
    for x, y in zip(metrics, values):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)

ax.set_xlabel("Evaluation Metric", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison by Metric", fontsize=14, fontweight="bold")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Model", fontsize=10, loc="upper right")

fig.tight_layout()
fig.savefig("model_metrics_lineplot.png", dpi=200, bbox_inches="tight")
print("saved: model_metrics_lineplot.png")
