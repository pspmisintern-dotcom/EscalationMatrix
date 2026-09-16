# Task: Apply cleaning to solution column + OpenAI answers for new causes

- [x] Add `load_dotenv()` in `app.py` so the `.env` (project root) API key is loaded
- [x] Apply `clean_text()` to cause + solution fields in `rag_service.py` (recommended_solution + similar case cause/solution)
- [x] Ensure OpenAI generates a solution for NEW causes (no high-confidence match) when API key is present
- [x] Apply `cleanCause()` to cause/solution in frontend similar-case cards (`App.jsx`)
- [x] Verify backend `/search` returns FULL cleaned solution + cause (no truncation, no ellipsis)
- [x] Verify frontend build (`npm run build`) — passes
- [x] Confirm `OPENAI_API_KEY` is present in `project/.env` (164 chars) and loads correctly
- [x] Confirm OpenAI integration is wired (code calls OpenAI when no high-confidence match)
- [ ] **Blocked externally**: OpenAI returns HTTP 429 `RateLimitError` — "You have no credits remaining. Add credits to continue using the API." The key is valid but the account has no quota. The backend gracefully falls back to the token-overlap heuristic until credits are added.

## Latest changes (proper English formatting, full text)
- Replaced truncation-based `clean_text`/`cleanCause` with a full-text English formatter:
  - Collapses whitespace/newlines into single spaces
  - Fixes spacing around punctuation and capitalizes sentences
  - Corrects common typos/abbreviations (e.g. "thki"→"thickness", "reqd"→"required", "obs"→"observed", "disaprch"→"dispatch", "recieved"→"received")
  - Returns the FULL text (no truncation, no ellipsis dots)
- **Fixed UI clipping**: The `.similar-card-title` and `.similar-card-value` had `-webkit-line-clamp: 3` + `overflow: hidden`, which clipped the problem/solution text to 3 lines in the UI even though the backend returned full text. Removed the line-clamp and changed `.similar-grid` `overflow-y: hidden` → removed so cards can grow to full height.
- Verified: `clean_text('Coating area thki as per Drawing size pulse. Pre-maching reqd')` → `'Coating area thickness as per Drawing size pulse. Pre-machining required'`
- Verified `/search` returns full 195-char solution and full cause sentence
- Frontend `npm run build` passes (26 modules)
- `App.jsx` applies `cleanCause()` to both `cause` and `solution` in the similar-case cards
- **Added arrow navigation** for the Similar Cases carousel:
  - `App.jsx`: added `useRef` (`similarRef`) + `scrollSimilar(direction)` to smoothly scroll the grid left/right by one card width
  - Wrapped the grid in `.similar-carousel` with circular left (`‹`) and right (`›`) arrow buttons
  - `App.css`: added `.similar-carousel`, `.similar-arrow` styles (rounded 40px buttons, hover state with indigo accent), and mobile responsive rules (stack vertically, center arrows)

## Notes
- `.env` exists at `project/.env` and now contains `OPENAI_API_KEY` (verified via `load_dotenv`).
- The OpenAI logic lives in `_build_generalized_summary()`; `load_dotenv()` was the missing wiring.
- The 429 error is an account/billing issue, not a code issue. Add credits on the OpenAI platform to enable AI-generated answers.

## Latency optimisations (AI response + streaming render)

- [x] `rag_service.py`: AI tuning constants + hard deadlines (`AI_FIRST_TOKEN_TIMEOUT`,
      `AI_TOTAL_TIMEOUT`), LRU answer cache (`AI_CACHE_SIZE`), daemon-thread Ollama stream with
      an explicit `stream.close()` so an abandoned generation cannot block the next search.
- [x] `rag_service.py`: `stream_answer()` AI phase rewritten — the grounded historical answer is
      emitted instantly, the AI answer streams in as an enhancement (`ai_start` / `text` /
      `ai_end`, `ai_skip` when the deadline passes, `ai_end {cached: true}` on cache replay).
- [x] `rag_service.py`: `RAGService.warmup()` pre-loads the embedder + FAISS index at startup.
- [x] `app.py`: `.env` loaded before the service imports; lifespan runs `service.warmup()` and
      `_warm_ollama()` (matched options + `keep_alive=-1`, so the model stays resident).
- [x] `embedding_service.py` / `retrieval_service.py`: thread-safe load locks +
      `local_files_only=True` fast path (no network round-trip when the model is already cached).
- [x] `App.jsx`: streamed tokens buffered and flushed once per animation frame
      (`createFrameBatcher`) instead of one React re-render per SSE event.
- [x] Prompt fixed: the model used to echo the literal placeholder example (`- point 1`,
      `1. step 1`). The format block now uses `<solution point>` / `<action step>` placeholders,
      explicitly tells the model not to repeat placeholder words, and the answer is real content
      grounded in the retrieved cases.
- [x] `project/.env`: AI knobs documented (`OLLAMA_MODEL`, `AI_MAX_CONTEXT_CASES`,
      `AI_MAX_FIELD_CHARS`, `AI_NUM_PREDICT`, `AI_NUM_CTX`, `AI_NUM_THREAD`, `AI_KEEP_ALIVE`,
      `AI_FIRST_TOKEN_TIMEOUT`, `AI_TOTAL_TIMEOUT`, `AI_CACHE_SIZE`) — see `README.md`.
- [x] Frontend verified: `npm run build` passes; `ai_start` / `ai_end` (incl. `truncated`,
      `cached`) / `ai_skip` all handled.

### Measured (CPU-only host: 4 cores, 8 GB RAM)

| Path | Baseline | After |
| --- | --- | --- |
| Novel cause (`/search/stream`, first content) | 34.99 s | ~0.5 s |
| Novel cause (AI answer complete) | 34.99 s | ~15.9 s (max 30 s, then `ai_skip`) |
| Novel cause, repeated (cache) | 34.99 s | **0.26 s** |
| Exact historical match | instant | instant |

- Verified AI output after the prompt fix (368 chars, `truncated: false`, real content):
  `Recommended Solution: - Check if the filter replacement instructions were followed correctly. …`
- Note: on this host the worst-case AI latency is dominated by Ollama's model load (TTFT measured
  at 0.3-38 s depending on whether the OS still has the 2 GB of weights in RAM). `keep_alive=-1`
  plus the startup warmup ping keeps them resident; the deadlines keep a slow model harmless.

## Disk space (resolved)

- C: was at ~0.25 GB free, which made Ollama page the 2 GB of model weights in and out
  (first-token latency measured between 0.3 s and 38 s) and made `ollama pull qwen2.5:1.5b`
  fail with "not enough space on the disk".
- Removed the two models the app never used: `llama3.2:latest` (2.0 GB) and `qwen3:8b` (5.2 GB).
  C: now has **6.99 GB free**; `qwen2.5:3b` and `nomic-embed-text` (both required at the time) are untouched,
  and the active model stays resident (`ollama ps` → `Forever`, ctx 1024).
- Decision (after the measurements below): the app now runs **`qwen2.5:1.5b`**
  (`OLLAMA_MODEL = "qwen2.5:1.5b"` in `project/.env`), which stays resident on this 8 GB
  machine and produced a complete AI answer in ~8.5-18 s instead of being skipped. The deadlines
  are unchanged (`AI_FIRST_TOKEN_TIMEOUT=15`, `AI_TOTAL_TIMEOUT=30`), the grounded historical
  answer is still instant, and `qwen2.5:3b` remains installed as a one-line swap back
  (`OLLAMA_MODEL = "qwen2.5:3b"`, better answers, needs more free RAM; `ollama rm qwen2.5:3b`
  frees another 1.9 GB if it is never wanted back).

### Model comparison (measured on this host, same prompt/options)

`qwen2.5:1.5b` was pulled for comparison (kept installed, **not** set in `.env`) using
`bench_ai_phase.py` with `OLLAMA_MODEL` overridden per run:

| | `qwen2.5:3b` | `qwen2.5:1.5b` |
| --- | --- | --- |
| Resident RAM (`ollama ps`) | 2.0 GB | 1.1 GB |
| Warm TTFT | 0.27-0.55 s | 0.27 s |
| Cold (weights evicted) TTFT | 38.5 s | 26.9 s |
| Generation rate | ~11.5 tok/s | ~11.5 tok/s |
| 80-token answer, warm | ~9-16 s | **8.5 s** |

- Generation speed is essentially the same: this CPU is limited by memory bandwidth / core count,
  not by the parameter count. The 1.5b's wins are RAM (~0.9 GB less, which is exactly what this
  8 GB machine lacks) and a faster cold load.
- Both models pinned at once (2.0 + 1.1 GB) put visible pressure on the box, so only one should be
  resident: `ollama stop qwen2.5:1.5b` was used to return to the 3b-only state after benchmarking.
- Note also: with ~2.5 GB of VS Code plus the node dev server and the embedder all running, free RAM
  drops to ~0.4 GB, which is what evicts the weights and causes the slow TTFT. Closing unused apps
  while using the app has the same effect as shrinking the model.


