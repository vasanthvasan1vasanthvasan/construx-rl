#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7860}"
IMAGE_NAME="${IMAGE_NAME:-construction-safety-env}"

echo "[1/4] Checking required files"
test -f openenv.yaml
test -f Dockerfile
test -f inference.py
test -f README.md

echo "[2/4] Building Docker image"
docker build -t "${IMAGE_NAME}" .

echo "[3/4] Starting container"
CONTAINER_ID="$(docker run -d -p "${PORT}:7860" "${IMAGE_NAME}")"
trap 'docker rm -f "${CONTAINER_ID}" >/dev/null 2>&1 || true' EXIT
sleep 5

echo "[4/4] Pinging environment"
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null
curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null

echo "Validation smoke test passed."
