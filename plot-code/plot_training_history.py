"""
plot_training_history.py

학습 이력 JSON 파일에서 epoch, accuracy, loss 값을 읽어
학습 정확도 및 학습 손실 그래프를 생성하는 스크립트입니다.

입력 JSON 예시:
{
    "epoch": [1, 2, 3, ...],
    "accuracy": [0.64, 0.75, 0.78, ...],
    "loss": [0.89, 0.64, 0.56, ...]
}

실행 예시:
    python plot_training_history.py
    python plot_training_history.py --input training_history_json_output.txt
    python plot_training_history.py --output-dir ./figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def plot_accuracy(epoch: list, accuracy: list, output_path: Path) -> None:
    """학습 정확도 그래프를 저장합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(epoch, accuracy, marker="o", linewidth=2, markersize=4)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_loss(epoch: list, loss: list, output_path: Path) -> None:
    """학습 손실 그래프를 저장합니다."""
    plt.figure(figsize=(8, 5))
    plt.plot(epoch, loss, marker="o", linewidth=2, markersize=4)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_combined(epoch: list, accuracy: list, loss: list, output_path: Path) -> None:
    """정확도와 손실을 하나의 그래프에 함께 저장합니다."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epoch, accuracy, marker="o", linewidth=2, markersize=4, label="Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epoch, loss, marker="s", linewidth=2, markersize=4, linestyle="--", label="Loss")
    ax2.set_ylabel("Loss")
    ax2.tick_params(axis="y")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("Training Accuracy and Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
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

    epoch = history["epoch"]
    accuracy = history["accuracy"]
    loss = history["loss"]

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
