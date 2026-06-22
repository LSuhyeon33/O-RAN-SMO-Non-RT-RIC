import matplotlib.pyplot as plt

# =========================
# 1. 성능 지표 입력
# =========================
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

baseline = [
    0.6623216601815823,
    0.5314764339186066,
    0.6623216601815823,
    0.5862976731267268
]

proposed = [
    0.862246000864678,
    0.8871088099919029,
    0.862246000864678,
    0.8511641600235963
]

# =========================
# 2. 꺾은선 그래프 생성
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    metrics,
    baseline,
    marker='o',
    linewidth=2,
    label='Baseline'
)

plt.plot(
    metrics,
    proposed,
    marker='o',
    linewidth=2,
    label='Proposed'
)

# =========================
# 3. 수치 표시
# =========================
for i, value in enumerate(baseline):
    plt.text(
        i,
        value + 0.015,
        f"{value:.3f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

for i, value in enumerate(proposed):
    plt.text(
        i,
        value + 0.015,
        f"{value:.3f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# =========================
# 4. 그래프 설정
# =========================
plt.title("Comparison of Detection Performance between Baseline and Proposed Architecture", fontsize=13)
plt.xlabel("Evaluation Metrics", fontsize=11)
plt.ylabel("Performance Score", fontsize=11)

plt.ylim(0.45, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.tight_layout()

# =========================
# 5. 저장 및 출력
# =========================
plt.savefig("baseline_vs_proposed_performance.png", dpi=300, bbox_inches='tight')
plt.show()