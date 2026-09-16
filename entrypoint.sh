#!/bin/sh
# Render entrypoint: starts Ollama beside the API in the same container.
# Render injects $PORT (default 8000 for local `docker run`).
set -e
PORT="${PORT:-8000}"

# Ollama daemon in the background.
(ollama serve > /tmp/ollama.log 2>&1 &)
sleep 3
# Idempotent: instant when the models are baked into the image, downloads them on first boot otherwise.
# Uses $OLLAMA_MODEL so swapping models via env needs no image change.
# Retried: on slow shared hosts `ollama serve` may still be starting.
for i in 1 2 3 4 5; do
  ollama pull "${OLLAMA_MODEL:-qwen2.5:1.5b}" > /tmp/ollama-pull.log 2>&1 && break || sleep 5
done || true
ollama pull nomic-embed-text >> /tmp/ollama-pull.log 2>&1 || true

cd /app/backend
# gunicorn + uvicorn worker: supports Render's $PORT, long AI streams
# (graceful-timeout must exceed AI_TOTAL_TIMEOUT=120 or the worker is killed
# mid-generation), single worker because the CPU-bound model serves one
# generation at a time anyway.
exec gunicorn app:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:$PORT" \
  --timeout 180 \
  --graceful-timeout 150 \
  --keep-alive 75
