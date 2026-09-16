# Backend + Ollama in ONE container for Render (Docker runtime).
# Build context / repo root == the `project/` folder.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    HF_HOME=/app/.cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/hf \
    OLLAMA_MODEL=qwen2.5:1.5b \
    AI_MAX_CONTEXT_CASES=1 \
    AI_MAX_FIELD_CHARS=70 \
    AI_NUM_PREDICT=80 \
    AI_NUM_CTX=1024 \
    AI_NUM_THREAD=0 \
    AI_KEEP_ALIVE=-1 \
    AI_FIRST_TOKEN_TIMEOUT=15 \
    AI_TOTAL_TIMEOUT=30 \
    AI_CACHE_SIZE=32

# curl (Ollama install) + libgomp1 (faiss-cpu OpenMP on slim images).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 procps zstd \
 && rm -rf /var/lib/apt/lists/*

# Ollama binary.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# CPU-only torch FIRST, so sentence-transformers does NOT pull the ~2.5 GB CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding weights into the image (avoids a ~90 MB download + slow start on boot).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Bake the Ollama models into the image so first boot doesn't download ~1.3 GB.
RUN (ollama serve > /tmp/ollama-build.log 2>&1 &) && sleep 5 && \
    ollama pull qwen2.5:1.5b && ollama pull nomic-embed-text && \
    pkill ollama || true

COPY backend ./backend
COPY data ./data
COPY models ./models
COPY entrypoint.sh ./entrypoint.sh
# Strip Windows CRLF (checkout on Windows breaks `#!/bin/sh`) and make executable.
RUN sed -i 's/\r$//' ./entrypoint.sh && chmod +x ./entrypoint.sh

EXPOSE 8000
CMD ["./entrypoint.sh"]
