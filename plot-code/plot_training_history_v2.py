"""
plot_training_history_v2.py

학습 이력 JSON 파일에서 epoch, accuracy, loss 값을 읽어
학습 정확도 및 학습 손실 그래프를 생성하는 스크립트입니다.

수정 반영 사항:
1. Accuracy 그래프는 초록색으로 표시합니다.
2. Accuracy, Loss 그래프의 y축 범위는 0.0~1.0으로 고정합니다.
3. epoch=0 지점을 accuracy=0, loss=1로 간주하여 그래프 앞에 추가합니다.

입력 JSON 예시:
{
    "epoch": [1, 2, 3, ...],
    "accuracy": [0.64, 0.75, 0.78, ...],
    "loss": [0.89, 0.64, 0.56, ...]
}

실행 예시:
    python plot_training_history_v2.py
    python plot_training_history_v2.py --input training_history_json_output.txt
    python plot_training_history_v2.py --output-dir ./figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ACCURACY_COLOR = "green"
Y_MIN = 0.0
Y_MAX = 1.0


def load_training_history(input_path: str) -> dict:
    """JSON 형식의 학습 이력 파일을 불러옵니다."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    required_keys = ["epoch", "accuracy", "loss"]
    missing_keys = [key for key in required_keys if key not in history]

    if missing_keys:
        raise KeyError(f"필수 키가 누락되었습니다: {missing_keys}")

    epoch_len = len(history["epoch"])
    if len(history["accuracy"]) != epoch_len or len(history["loss"]) != epoch_len:
        raise ValueError("epoch, accuracy, loss의 데이터 길이가 서로 다릅니다.")

    return history


def add_initial_epoch(history: dict) -> tuple[list, list, list]:
    """epoch=0, accuracy=0, loss=1 지점을 그래프 데이터 앞에 추가합니다."""
    epoch = list(history["epoch"])
    accuracy = list(history["accuracy"])
    loss = list(history["loss"])

    # 이미 epoch=0이 포함되어 있으면 중복 추가하지 않고 값을 요구사항에 맞게 보정합니다.
    if epoch and epoch[0] == 0:
        accuracy[0] = 0.0
        loss[0] = 1.0
    else:
        epoch.insert(0, 0)
        accuracy.insert(0, 0.0)
        loss.insert(0, 1.0)

    return epoch, accuracy, loss


def plot_accuracy(epoch: list, accuracy: list, output_path: Path) -> None:
    """학습 정확도 그래프를 저장합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch,
        accuracy,
        marker="o",
        linewidth=2,
        markersize=4,
        color=ACCURACY_COLOR,
        label="Accuracy",
    )
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(Y_MIN, Y_MAX)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_loss(epoch: list, loss: list, output_path: Path) -> None:
    """학습 손실 그래프를 저장합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(epoch, loss, marker="o", linewidth=2, markersize=4, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(Y_MIN, Y_MAX)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_combined(epoch: list, accuracy: list, loss: list, output_path: Path) -> None:
    """정확도와 손실을 하나의 그래프에 함께 저장합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch,
        accuracy,
        marker="o",
        linewidth=2,
        markersize=4,
        color=ACCURACY_COLOR,
        label="Accuracy",
    )
    # plt.plot(epoch, loss, marker="s", linewidth=2, markersize=4, linestyle="--", label="Loss")
    plt.plot(epoch, loss, marker="s", linewidth=2, markersize=4, label="Loss")

    plt.title("Training Accuracy and Loss - Transformer")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(Y_MIN, Y_MAX)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="학습 손실과 정확도 그래프를 생성합니다."
    )
    parser.add_argument(
        "--input",
        default="training_history_json_output.txt",
        help="학습 이력 JSON 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        default="training_plots",
        help="그래프 이미지 저장 폴더",
    )
    args = parser.parse_args()

    history = load_training_history(args.input)
    epoch, accuracy, loss = add_initial_epoch(history)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = output_dir / "training_accuracy.png"
    loss_path = output_dir / "training_loss.png"
    combined_path = output_dir / "training_accuracy_loss.png"

    plot_accuracy(epoch, accuracy, accuracy_path)
    plot_loss(epoch, loss, loss_path)
    plot_combined(epoch, accuracy, loss, combined_path)

    print("그래프 저장 완료")
    print(f"- 정확도 그래프: {accuracy_path}")
    print(f"- 손실 그래프: {loss_path}")
    print(f"- 통합 그래프: {combined_path}")


if __name__ == "__main__":
    main()
