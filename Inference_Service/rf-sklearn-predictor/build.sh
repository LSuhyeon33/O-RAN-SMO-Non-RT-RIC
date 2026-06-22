#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

IMAGE_NAME="${IMAGE_NAME:-rf-sklearn-predictor:dev}"

echo "[build] $IMAGE_NAME"
sudo buildctl --addr=nerdctl-container://buildkitd build \
    --frontend dockerfile.v0 \
    --opt filename=Dockerfile \
    --local dockerfile=. \
    --local context=. \
    --output type=oci,name="$IMAGE_NAME" | sudo nerdctl --namespace k8s.io load
