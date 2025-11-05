#!/usr/bin/env bash
set -euo pipefail

: "${SMARTAI_VIDEO_DIR:=/app/videos}"
: "${HF_HOME:=/app/.cache/huggingface}"
: "${TRANSFORMERS_CACHE:=/app/.cache/huggingface}"
: "${SENTENCE_TRANSFORMERS_HOME:=/app/.cache/sentence-transformers}"
: "${PLAYWRIGHT_BROWSERS_PATH:=/root/.cache/ms-playwright}"

echo "[init] SMARTAI_VIDEO_DIR=${SMARTAI_VIDEO_DIR}"
echo "[init] HF_HOME=${HF_HOME}"
echo "[init] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[init] SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME}"
echo "[init] PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH}"

mkdir -p "${SMARTAI_VIDEO_DIR}"          "${HF_HOME}"          "${SENTENCE_TRANSFORMERS_HOME}"          "${PLAYWRIGHT_BROWSERS_PATH}"

if ! find "${PLAYWRIGHT_BROWSERS_PATH}" -maxdepth 2 -type f -name "headless_shell" | grep -q headless_shell; then
  echo "[init] Playwright browsers not found. Installing Chromium..."
  python -m playwright install chromium || python -m playwright install chromium
else
  echo "[init] Playwright browsers already present."
fi

exec xvfb-run -a python -m uvicorn main:app --host 0.0.0.0 --port 8001

