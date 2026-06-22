import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 데이터 입력
# =========================
data = {
    "batch_size": [1, 10, 50, 100, 150, 200, 250, 300],
    "baseline": [
        3097.9176572489087,
        320.38852611207403,
        67.37128088995814,
        39.94628659100272,
        30.37609477294609,
        25.29074377613142,
        22.657129562227055,
        20.443913949886337
    ],
    "proposed": [
        405.7776442680042,
        80.80863255797885,
        50.331200893968344,
        45.861927377060056,
        44.15525712794624,
        42.699497291003354,
        42.699497291003354,
        42.023471558932215
    ]
}

df = pd.DataFrame(data)

# 단축률 계산: Baseline 대비 Proposed가 얼마나 감소했는지
df["reduction_rate"] = ((df["baseline"] - df["proposed"]) / df["baseline"]) * 100

print(df)

# =========================
# 2. 그래프 설정
# =========================
# plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 기준 한글 폰트
# plt.rcParams["axes.unicode_minus"] = False

# 글자 크기 설정 (한 곳에서 조절)
TITLE_FONTSIZE = 18   # 제목
LABEL_FONTSIZE = 16   # 가로축/세로축 라벨
TICK_FONTSIZE = 14    # 눈금(성능 지표) 숫자
LEGEND_FONTSIZE = 14  # 범례

# =========================
# 3. 지연시간 비교 그래프
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    df["batch_size"],
    df["baseline"],
    marker="o",
    label="Baseline"
)

plt.plot(
    df["batch_size"],
    df["proposed"],
    marker="s",
    label="Proposed"
)

plt.xlabel("Batch Size", fontsize=LABEL_FONTSIZE)
plt.ylabel("Inference Latency (ms)", fontsize=LABEL_FONTSIZE)
plt.title("Inference Latency by Batch Size", fontsize=TITLE_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("latency_comparison.png", dpi=300)
plt.show()

# =========================
# 4. 단축률 그래프
# =========================
plt.figure(figsize=(8, 5))

plt.bar(
    df["batch_size"].astype(str),
    df["reduction_rate"]
)

plt.xlabel("Batch Size", fontsize=LABEL_FONTSIZE)
plt.ylabel("Reduction Rate (%)", fontsize=LABEL_FONTSIZE)
plt.title("Inference Latency Reduction Rate by Batch Size", fontsize=TITLE_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("latency_reduction_rate.png", dpi=300)
plt.show()