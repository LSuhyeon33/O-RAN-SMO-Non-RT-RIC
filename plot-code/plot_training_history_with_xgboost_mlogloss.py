#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training history graphs.

지원 형식
1) 일반 딥러닝 학습 이력:
   {
     "epoch": [...],
     "accuracy": [...],
     "loss": [...]
   }

2) XGBoost mlogloss 학습 이력:
   {
     "epoch": [...],
     "train_mlogloss": [...]
   }

기능
- accuracy 그래프: 초록색
- accuracy/loss 그래프 y축 범위: 0.0 ~ 1.0
- accuracy/loss 데이터에는 epoch=0, accuracy=0, loss=1 추가
- XGBoost train_mlogloss 그래프 생성
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(input_path: str | Path) -> dict:
    """JSON 또는 txt 확장자의 학습 이력 파일을 읽는다."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """출력 폴더를 생성하고 Path 객체로 반환한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def add_epoch_zero_for_accuracy_loss(history: dict) -> tuple[list, list, list]:
    """
    accuracy/loss 그래프용으로 epoch=0 지점을 추가한다.

    epoch=0:
    - accuracy=0
    - loss=1
    """
    epochs = list(history["epoch"])
    accuracy = list(history["accuracy"])
    loss = list(history["loss"])

    if len(epochs) != len(accuracy) or len(epochs) != len(loss):
        raise ValueError("epoch, accuracy, loss 길이가 서로 다릅니다.")

    if len(epochs) == 0:
        raise ValueError("epoch 데이터가 비어 있습니다.")

    if epochs[0] != 0:
        epochs = [0] + epochs
        accuracy = [0.0] + accuracy
        loss = [1.0] + loss

    return epochs, accuracy, loss


def plot_accuracy(history: dict, output_dir: Path) -> Path:
    """Accuracy 그래프를 초록색으로 저장한다."""
    epochs, accuracy, _ = add_epoch_zero_for_accuracy_loss(history)

    output_path = output_dir / "training_accuracy.png"

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, marker="o", linewidth=2, color="green", label="Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_loss(history: dict, output_dir: Path) -> Path:
    """Loss 그래프를 저장한다."""
    epochs, _, loss = add_epoch_zero_for_accuracy_loss(history)

    output_path = output_dir / "training_loss.png"

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, marker="o", linewidth=2, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_accuracy_loss(history: dict, output_dir: Path) -> Path:
    """Accuracy와 Loss를 하나의 그래프로 저장한다."""
    epochs, accuracy, loss = add_epoch_zero_for_accuracy_loss(history)

    output_path = output_dir / "training_accuracy_loss.png"

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, marker="o", linewidth=2, color="green", label="Accuracy")
    plt.plot(epochs, loss, marker="s", linewidth=2, label="Loss")
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_xgboost_mlogloss(history: dict, output_dir: Path) -> Path:
    """
    XGBoost multi-class log loss 그래프를 저장한다.

    입력 파일 예시:
    {
      "epoch": [1, 2, 3, ...],
      "train_mlogloss": [1.55, 1.38, 1.24, ...]
    }
    """
    if "epoch" not in history:
        raise KeyError("history에 'epoch' 키가 없습니다.")

    if "train_mlogloss" not in history:
        raise KeyError("history에 'train_mlogloss' 키가 없습니다.")

    epochs = list(history["epoch"])
    train_mlogloss = list(history["train_mlogloss"])

    if len(epochs) != len(train_mlogloss):
        raise ValueError("epoch와 train_mlogloss 길이가 서로 다릅니다.")

    output_path = output_dir / "xgboost_train_mlogloss.png"

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        train_mlogloss,
        marker="o",
        linewidth=2,
        label="Train mlogloss",
    )
    plt.title("Training mlogloss - XGBoost")
    plt.xlabel("Estimators")
    plt.ylabel("mlogloss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy/loss graphs or XGBoost mlogloss graph from a JSON training history file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to training history JSON/txt file.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_plots",
        help="Directory where graph images will be saved.",
    )
    args = parser.parse_args()

    history = load_history(args.input)
    output_dir = ensure_output_dir(args.output_dir)

    saved_files = []

    # 일반 accuracy/loss 학습 이력
    if {"epoch", "accuracy", "loss"}.issubset(history.keys()):
        saved_files.append(plot_accuracy(history, output_dir))
        saved_files.append(plot_loss(history, output_dir))
        saved_files.append(plot_accuracy_loss(history, output_dir))

    # XGBoost mlogloss 학습 이력
    if {"epoch", "train_mlogloss"}.issubset(history.keys()):
        saved_files.append(plot_xgboost_mlogloss(history, output_dir))

    if not saved_files:
        available_keys = ", ".join(history.keys())
        raise ValueError(
            "지원되는 학습 이력 형식이 아닙니다. "
            "필요 키: {'epoch','accuracy','loss'} 또는 {'epoch','train_mlogloss'} / "
            f"현재 키: {available_keys}"
        )

    print("Saved graph files:")
    for file_path in saved_files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
