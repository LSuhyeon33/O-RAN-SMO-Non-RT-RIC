import matplotlib.pyplot as plt

baseline = 2174.42
proposed = 607.92

reduction = (baseline - proposed) / baseline * 100

# 글자 크기 설정 (한 곳에서 조절)
TITLE_FONTSIZE = 18   # 제목
LABEL_FONTSIZE = 16   # 세로축 라벨
TICK_FONTSIZE = 14    # 눈금(가로축/세로축 성능 지표) 숫자
VALUE_FONTSIZE = 14   # 막대 위 값 라벨
REDUCTION_FONTSIZE = 16  # 감소율 텍스트

fig, ax = plt.subplots(figsize=(6,5))

bars = ax.bar(
    ["Baseline", "Proposed"],
    [baseline, proposed],
    color=["#A6A6A6", "#005241"]
)

for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x()+bar.get_width()/2,
        h,
        f"{h:.2f}",
        ha='center',
        va='bottom',
        fontsize=VALUE_FONTSIZE
    )

# 화살표 위치
arrow_y = baseline * 1.10

ax.annotate(
    "",
    xy=(1, proposed * 1.15),
    xytext=(0, baseline),
    arrowprops=dict(
        arrowstyle="->",
        lw=2.5,
        color="#C00000"
    )
)

# 감소율 텍스트
ax.text(
    0.5,
    arrow_y * 1.03,
    f"-{reduction:.1f}%",
    ha="center",
    va="bottom",
    fontsize=REDUCTION_FONTSIZE,
    fontweight="bold",
    color="#C00000"
)

# 상단 여백 확보
ax.set_ylim(0, baseline * 1.35)

ax.set_ylabel("Memory Usage (MB)", fontsize=LABEL_FONTSIZE)
ax.set_title("Memory Usage Comparison", fontsize=TITLE_FONTSIZE)
ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

plt.tight_layout()
plt.savefig("memory_usage_arrow.png", dpi=300)
plt.show()