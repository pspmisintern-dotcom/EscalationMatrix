# Escalation Management RAG

## Overview
This project provides a retrieval-augmented escalation assistant that reads Excel escalation history, embeds the causes, stores them in FAISS, and exposes a FastAPI backend for semantic search and save workflows.

## Backend
Run:

```bash
cd project/backend
uvicorn app:app --reload --port 8000
```

## Frontend
Run:

```bash
cd project/frontend
npm install
npm start
```

## Performance & AI tuning (CPU-only host)

The app targets a CPU-only machine (4 cores / 8 GB RAM in this deployment), where
the *model load* and *output length* dominate latency. The following keeps first
content fast and bounds every AI wait:

- **Startup warmup** — `RAGService.warmup()` (called from the FastAPI lifespan) loads the
  embedding model + FAISS index in a background thread; `warm_ollama_async()` pre-loads the
  Ollama model with `keep_alive=-1` (kept resident). A search that arrives mid-warmup waits
  on the load lock instead of loading a second copy. A cold embedder load costs ~20-30 s,
  which is now paid at boot rather than on the user's first query.
- **`/search/stream`** emits the grounded historical answer immediately (~0.1 s, chunked into
  small word groups) and then streams the local-model answer as an *enhancement*:
  - `ai_start` / `text` (AI tokens) / `ai_end` events; `ai_end` carries `cached: true` on a
    cache hit, and `ai_skip` is sent when the model was too slow and the AI part was skipped.
  - The AI phase is bounded by `AI_FIRST_TOKEN_TIMEOUT` (give up if the first token is that
    late) and `AI_TOTAL_TIMEOUT` (hard cap for the whole generation). Abandoned generations
    are explicitly closed so they cannot keep the single CPU-bound model busy for the next user.
  - Complete answers are cached (LRU, `AI_CACHE_SIZE`), so repeating a cause is instant.
- **Frontend** (`App.jsx`) buffers streamed tokens and repaints once per animation frame
  (`createFrameBatcher`), instead of re-rendering per token.
- **Tuning knobs** — all optional, read by `services/rag_service.py` at import time from
  `project/.env` (see `.env` for the documented list): `OLLAMA_MODEL`, `AI_MAX_CONTEXT_CASES`,
  `AI_MAX_FIELD_CHARS`, `AI_NUM_PREDICT`, `AI_NUM_CTX`, `AI_NUM_THREAD`, `AI_KEEP_ALIVE`,
  `AI_FIRST_TOKEN_TIMEOUT`, `AI_TOTAL_TIMEOUT`, `AI_CACHE_SIZE`, plus
  `OLLAMA_WARMUP_TIMEOUT` (startup warmup request budget, default 300 s).
  Lower `AI_NUM_PREDICT` on slower hosts, or use a smaller model
  (e.g. `qwen2.5:1.5b`) for roughly 2x faster generation.
- **No keep-warm heartbeat** — a 1-token heartbeat (every 15 s and every 30 s) was
  tried to stop the OS paging weights out, but made no measurable difference
  (first-token latency stayed ~9.5-11 s either way), so it was removed rather than
  left in as background CPU load. `keep_alive=-1` plus the startup warmup is what
  keeps the model resident.

Measured on this host, baseline → after: novel cause **34.99 s → ~16-19 s complete**
(~10 s to first AI token, grounded historical answer still instant at ~0.5 s),
exact-match cause instant in both, repeat cause near-instant (AI answer cache).
Helper scripts used for those numbers (kept outside the app for reference):
`probe_ai.py`, `bench_ai_phase.py`, `perf_after.py`, `probe_multi.py`.
## Notes

- The system uses the local model `all-MiniLM-L6-v2` and stores the FAISS index in `project/models/`.
- **Accurate data loading**: the escalation log stores one row per escalation level (L1, L2, …).
  The loader merges those rows into ONE record per escalation — the cause is cleaned, the full
  effort/action trail across all levels is preserved (`L1 action | L2 action`), the final
  resolution (Solution column) is never lost, status reflects the final state, and metadata
  (escalation ID, department, customer, material, KP no., level trail, latest timestamp) is kept.
- **Hybrid retrieval**: FAISS cosine search (over cause + actions + solution) is re-ranked with a
  keyword-overlap score (72% semantic / 28% keyword). Reported similarity is the honest combined
  score — scores ≥ 0.62 are treated as a direct match, otherwise the local Ollama model generates
  a grounded recommendation.
- **Safe saves**: `/save` appends the new escalation in the file's ORIGINAL schema (auto-assigning
  the next escalation ID) and refreshes the index — the file's columns and existing rows are never
  rewritten.

