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
# 2. 그래프 생성
# =========================
plt.figure(figsize=(9, 5))

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
        value - 0.035,
        f"{value:.3f}",
        ha='center',
        va='top',
        fontsize=10
    )

for i, value in enumerate(proposed):
    plt.text(
        i,
        value + 0.025,
        f"{value:.3f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# =========================
# 4. 증가량 화살표 표시
# =========================
for i, (b, p) in enumerate(zip(baseline, proposed)):
    increase = p - b
    increase_rate = (increase / b) * 100

    # Baseline에서 Proposed로 향하는 화살표
    plt.annotate(
        '',
        xy=(i, p - 0.01),
        xytext=(i, b + 0.01),
        arrowprops=dict(
            arrowstyle='->',
            linewidth=1.8,
            color='red'
        )
    )

    # 증가량 텍스트
    plt.text(
        i + 0.08,
        (b + p) / 2,
        f"+{increase:.3f}\n({increase_rate:.1f}%)",
        ha='left',
        va='center',
        fontsize=9,
        fontweight='bold',
        color='red'
    )

# =========================
# 5. 그래프 설정
# =========================
plt.title("Performance Improvement from Baseline to Proposed Architecture", fontsize=13)
plt.xlabel("Evaluation Metrics", fontsize=11)
plt.ylabel("Performance Score", fontsize=11)

plt.ylim(0.45, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.tight_layout()

# =========================
# 6. 저장 및 출력
# =========================
plt.savefig("baseline_vs_proposed_performance_with_arrows.png", dpi=300, bbox_inches='tight')
plt.show()