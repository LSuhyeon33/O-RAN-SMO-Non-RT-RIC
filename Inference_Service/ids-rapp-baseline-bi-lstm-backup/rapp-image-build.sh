#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# 학습 산출물이 있어야 베이크 가능 (Proposed 와 달리 모델을 이미지에 직접 넣는다)
[[ -f model/model.keras && -f model/scaler.pkl ]] || {
  echo "[error] model/ 비어있음 — 먼저 'python train/train.py' 실행"; exit 1; }

IMAGE_NAME="${IMAGE_NAME:-ids-rapp-baseline-bi-lstm:dev}"

# build context = baseline 루트(.) — Dockerfile 이 rapp/app, train/train.py, model/ 을 모두 베이크
echo "[build] $IMAGE_NAME"
sudo buildctl --addr=nerdctl-container://buildkitd build \
    --frontend dockerfile.v0 \
    --opt filename=Dockerfile \
    --local dockerfile=rapp \
    --local context=. \
    --output type=oci,name="$IMAGE_NAME" | sudo nerdctl --namespace k8s.io load
