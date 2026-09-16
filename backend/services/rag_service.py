from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from services.excel_loader import ExcelLoader
from services.retrieval_service import RetrievalService
from services.text_cleaning import clean_text, summarize_text  # noqa: F401  (re-exported for tests)


# Text-cleaning helpers (clean_text, summarize_text, tokenize) now live in
# services/text_cleaning.py so the loader, retriever, and this module all
# normalize text identically (index and queries are cleaned the same way).

# ---------------------------------------------------------------------------
# Local AI (Ollama) tuning.
#
# This deployment runs the model on the CPU, where the model load time
# (TTFT) and output length dominate latency. Every knob can be overridden from
# the project `.env` file, and all AI work is wrapped in hard deadlines so a
# slow/cold model can never stall a request again (the novel path used to stay
# open for 35-80 s with no bound at all).
#
# Measured on the current host (4 cores, 8 GB RAM): first token 0.3 s when the
# model is resident in RAM, but ~15-38 s when the OS has to page the 2 GB of
# weights back in from disk; generation runs at ~4-12 tokens/second. Hence: the
# grounded historical answer is emitted instantly, the AI answer is a bounded
# enhancement, and repeated causes are served from the answer cache.
# ---------------------------------------------------------------------------
AI_MAX_CONTEXT_CASES = int(os.getenv("AI_MAX_CONTEXT_CASES", "1"))
AI_MAX_FIELD_CHARS = int(os.getenv("AI_MAX_FIELD_CHARS", "70"))
AI_NUM_PREDICT = int(os.getenv("AI_NUM_PREDICT", "80"))
AI_NUM_CTX = int(os.getenv("AI_NUM_CTX", "1024"))
AI_NUM_THREAD = int(os.getenv("AI_NUM_THREAD", "0")) or (os.cpu_count() or 4)
# `-1` keeps the model resident, so the cold load is paid only once.
AI_KEEP_ALIVE = os.getenv("AI_KEEP_ALIVE", "-1")
# First token must arrive inside this budget, else the AI refinement is skipped
# and the already-displayed data-driven answer is kept.
AI_FIRST_TOKEN_TIMEOUT = float(os.getenv("AI_FIRST_TOKEN_TIMEOUT", "15"))
# Hard cap for the whole generation (instant answer is on screen meanwhile).
AI_TOTAL_TIMEOUT = float(os.getenv("AI_TOTAL_TIMEOUT", "30"))
AI_CACHE_SIZE = int(os.getenv("AI_CACHE_SIZE", "32"))

# --- Cloud AI (OpenAI) -------------------------------------------------------
# Optional fast path alongside the local Ollama model. When OPENAI_API_KEY is
# set (local `.env` or Render env var), novel causes try OpenAI FIRST (fast,
# no local RAM cost) and fall back to Ollama, then to the heuristic.
# AI_PROVIDER selects the chain: "auto" (default: openai -> ollama ->
# heuristic), "openai" (cloud only), "ollama" (local only), "off" (no AI).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
try:
    OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))
except ValueError:
    OPENAI_TIMEOUT = 30.0
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").strip().lower() or "auto"

_AI_WARMUP_TOKENS = 8  # startup warmup length: enough decode steps to page in weights

# Sentinel pushed by the Ollama worker thread when the stream is over.
_STREAM_END = object()

_AI_CACHE: dict[str, str] = {}
_AI_CACHE_LOCK = threading.Lock()


def ollama_keep_alive() -> int | str:
    """Ollama accepts plain seconds (int) or a duration string like "30m".

    "-1" must be sent as the NUMBER -1 ("forever"); Ollama rejects the string
    form because `time.ParseDuration("-1")` has no unit.
    """
    raw = str(AI_KEEP_ALIVE).strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _shorten(value: Any, limit: int = AI_MAX_FIELD_CHARS) -> str:
    """Collapse whitespace and cap a field so AI prompts stay small."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _openai_available() -> bool:
    """True when the cloud OpenAI path may be used for this request."""
    if AI_PROVIDER in ("off", "ollama"):
        return False
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _ollama_available() -> bool:
    """True when the local Ollama path may be used for this request."""
    return AI_PROVIDER not in ("off", "openai")


def _openai_answer(cause_query: str, context_cases: list[dict[str, Any]]) -> str:
    """One bounded OpenAI call reusing the Ollama prompt shape.

    Returns the stripped answer text, or "" when the key is missing, the
    package is missing, AI_PROVIDER disallows it, or the call fails/times
    out. Never raises — callers fall through to Ollama, then heuristic.
    """
    if AI_PROVIDER in ("off", "ollama"):
        return ""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""
    system_prompt, user_prompt = _build_ai_prompts(cause_query, context_cases)
    try:
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # quota/offline/slow -> Ollama is next
        print(f"[rag] OpenAI failed, falling back to Ollama: {exc!r}", file=sys.stderr, flush=True)
        return ""


def _ai_cache_get(cache_key: str) -> str:
    with _AI_CACHE_LOCK:
        return _AI_CACHE.get(cache_key, "")


def _ai_cache_put(cache_key: str, value: str) -> None:
    if not value:
        return
    with _AI_CACHE_LOCK:
        while len(_AI_CACHE) >= AI_CACHE_SIZE:
            _AI_CACHE.pop(next(iter(_AI_CACHE)), None)
        _AI_CACHE[cache_key] = value


def _start_ollama_stream(
    system_prompt: str,
    user_prompt: str,
) -> tuple[queue.Queue, threading.Event]:
    """Run a streaming Ollama chat in a daemon thread; return its token queue.

    The blocking HTTP stream lives in a worker thread so the caller can enforce
    its own deadlines instead of waiting minutes for a CPU-bound model. The
    stream always ends with the `_STREAM_END` sentinel, success or not.
    """
    tokens: queue.Queue = queue.Queue()
    cancelled = threading.Event()

    def _worker() -> None:
        stream = None
        try:
            import ollama

            client = ollama.Client(timeout=AI_TOTAL_TIMEOUT + 15)
            stream = client.chat(
                model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": AI_NUM_PREDICT,
                    "num_ctx": AI_NUM_CTX,
                    "num_thread": AI_NUM_THREAD,
                },
                keep_alive=ollama_keep_alive(),  # keeps the model loaded
                stream=True,
            )
            for chunk in stream:
                if cancelled.is_set():
                    break
                piece = (chunk.get("message") or {}).get("content", "") if chunk else ""
                if piece:
                    tokens.put(piece)
        except Exception as exc:  # noqa: BLE001  (the model must never break a search)
            # A failure here is swallowed on purpose — the data-driven answer is
            # always available — but it is logged, because a silent failure just
            # looks like a mysterious "ai_skip" to whoever is debugging latency.
            print(f"[rag] Ollama stream failed: {exc!r}", file=sys.stderr, flush=True)
        finally:
            # Closing the iterator aborts the HTTP stream, so a request we gave
            # up on does not keep the (single) CPU-bound model busy and block
            # the next user's search.
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            tokens.put(_STREAM_END)

    threading.Thread(target=_worker, daemon=True, name="ollama-stream").start()
    return tokens, cancelled


def _iter_openai_tokens(
    cause_query: str,
    context_cases: list[dict[str, Any]],
    completed: list | None = None,
) -> Iterator[str]:
    """Yield OpenAI tokens as they stream in, within OPENAI_TIMEOUT.

    Behaves like the Ollama token iterator: yields nothing when the key is
    missing, AI_PROVIDER disallows the cloud path, the package is missing, or
    nothing arrives inside the timeout. `completed` gains True only when the
    model finished on its own. Never raises — callers fall through to Ollama.
    """
    if not _openai_available():
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return
    try:
        from openai import OpenAI
    except Exception:
        return
    system_prompt, user_prompt = _build_ai_prompts(cause_query, context_cases)
    pieces: queue.Queue = queue.Queue()
    done = threading.Event()

    def _worker() -> None:
        stream = None
        try:
            client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=300,
                stream=True,
            )
            for chunk in stream:
                if done.is_set():
                    break
                try:
                    delta = (chunk.choices[0].delta.content if chunk.choices else "") or ""
                except Exception:
                    delta = ""
                if delta:
                    pieces.put(delta)
        except Exception as exc:  # quota/offline/slow -> Ollama is next
            print(f"[rag] OpenAI stream failed, falling back to Ollama: {exc!r}", file=sys.stderr, flush=True)
        finally:
            try:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()
            except Exception:
                pass
            pieces.put(_STREAM_END)

    threading.Thread(target=_worker, daemon=True, name="openai-stream").start()
    deadline = time.monotonic() + OPENAI_TIMEOUT
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                piece = pieces.get(timeout=remaining)
            except queue.Empty:
                break
            if piece is _STREAM_END:
                if completed is not None:
                    completed.append(True)
                return
            yield piece
    finally:
        done.set()  # stop consuming the worker stream


def _iter_ai_tokens(
    cause_query: str,
    context_cases: list[dict[str, Any]],
    completed: list | None = None,
) -> Iterator[str]:
    """Yield AI tokens, stopping as soon as a deadline is exceeded.

    Chain: OpenAI (cloud, fast) -> Ollama (local) -> nothing. Each stage is
    bounded by its own timeout; skipped stages simply yield nothing.

    Nothing is yielded when no model answers in time (cold model / Ollama
    down / OpenAI quota): callers then keep the instant data-driven
    recommendation instead of making the user wait on the model.
    `completed` (a one-item list, used as an out-parameter) is set to True only
    when a model finished the answer on its own.
    """
    if _openai_available():
        openai_done: list = []
        yielded_any = False
        for piece in _iter_openai_tokens(cause_query, context_cases, openai_done):
            yielded_any = True
            yield piece
        if yielded_any:
            if completed is not None and openai_done:
                completed.append(True)
            return
        # OpenAI produced nothing (quota/offline/timeout) -> try local Ollama.
    if not _ollama_available():
        return
    system_prompt, user_prompt = _build_ai_prompts(cause_query, context_cases)
    tokens, cancelled = _start_ollama_stream(system_prompt, user_prompt)
    started = False
    deadline = time.monotonic() + AI_TOTAL_TIMEOUT
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = remaining if started else min(AI_FIRST_TOKEN_TIMEOUT, remaining)
            try:
                piece = tokens.get(timeout=wait)
            except queue.Empty:
                break  # first-token timeout (cold/offline) or total timeout
            if piece is _STREAM_END:
                if completed is not None:
                    completed.append(True)
                return
            started = True
            yield piece
    finally:
        cancelled.set()  # stop consuming the worker stream


def _ai_answer(
    cause_query: str,
    context_cases: list[dict[str, Any]],
) -> str:
    """Return a COMPLETE AI answer for the cause, or "" when unavailable.

    Used by the non-streaming `/search` endpoint, which cannot re-render
    progressively — a half-finished answer would be worse than the grounded
    historical one, so partial output is rejected. Answers are cached per
    cause, so asking the same thing twice is instant.
    """
    cache_key = cause_query.strip().lower()
    cached = _ai_cache_get(cache_key)
    if cached:
        return cached

    completed: list = []
    answer = "".join(_iter_ai_tokens(cause_query, context_cases, completed)).strip()
    if not answer or not completed:
        return ""
    _ai_cache_put(cache_key, answer)
    return answer





def _build_ai_prompts(cause_query: str, context_cases: list[dict[str, Any]]) -> tuple[str, str]:
    """Build the system + user prompts for the local Ollama model.

    The prompt requests a POINT-WISE, SYSTEMATIC answer (bullet points for the
    solution + numbered steps for the effort/action) grounded strictly in the
    historical cases passed as context.
    """
    # Only the two closest cases are used, and every field is capped, so the
    # prompt stays small: the model runs on the CPU, where prompt size and
    # output length are the dominant latency costs.
    truncated = context_cases[:AI_MAX_CONTEXT_CASES]
    if truncated:
        # Compact one-line context: prompt evaluation costs ~40 tokens/second on
        # this CPU, so every character here is paid for on the first AI token.
        historical_context = chr(10).join(
            f"- cause: {_shorten(c.get('cause'))}"
            f" | fix: {_shorten(c.get('solution'))}"
            f" | done: {_shorten(c.get('effort_taken'))}"
            for c in truncated
        )
    else:
        historical_context = "No similar historical cases found."

    system_prompt = (
        "You are an escalation assistant for a manufacturing plant. Using ONLY the "
        "case below, give a short solution and the next actions. Never invent "
        "case-specific details."
    )
    user_prompt = (
        "Cause: " + cause_query
        + chr(10)
        + "Cases:"
        + chr(10)
        + historical_context
        + chr(10) * 2
        + "Write real content from the case above, never placeholder words. At most "
        + "3 bullets and 3 steps, each line at most 12 words, exactly like this:"
        + chr(10)
        + "Recommended Solution:"
        + chr(10)
        + "- <point>"
        + chr(10)
        + "Recommended Effort/Action:"
        + chr(10)
        + "1. <step>"
    )
    return system_prompt, user_prompt


def warm_ollama_async() -> None:
    """Load the model AND push a realistic prompt through it, in the background.

    Ollama reports a model as "loaded" after the first tiny request, but the
    first *real* search still pays a multi-second page-in of the weights on a
    RAM-tight host. Measured with qwen2.5:1.5b on this machine (Ollama's own
    metrics): the first realistic prompt spent 5.6 s in prompt evaluation while
    the next spent 0.11 s, with `load_duration` 0.00 s in both cases — that is
    the cost that used to land on the user's first search and push its first
    token past `AI_FIRST_TOKEN_TIMEOUT` (i.e. a skipped AI block).

    Sending the real prompt shape plus a few generated tokens at startup moves
    that cost off the first search. It runs on a daemon thread, so `/health`
    answers immediately and searches work whether or not this has finished.

    (A 1-token "keep-warm" heartbeat was tried here too, every 15 s and every
    30 s, to stop the OS from paging the weights out. It made no measurable
    difference — first-token latency stayed at ~9.5-11 s either way — so it was
    removed rather than left in as background CPU load.)
    """

    def _warm() -> None:
        try:
            import ollama

            system_prompt, user_prompt = _build_ai_prompts("startup warmup", [])
            client = ollama.Client(timeout=float(os.getenv("OLLAMA_WARMUP_TIMEOUT", "300")))
            client.chat(
                model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    # A few tokens, so the decode path is exercised as well.
                    "num_predict": _AI_WARMUP_TOKENS,
                    "num_ctx": AI_NUM_CTX,
                    "num_thread": AI_NUM_THREAD,
                },
                keep_alive=ollama_keep_alive(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] Ollama warmup skipped: {exc!r}", file=sys.stderr, flush=True)

    threading.Thread(target=_warm, daemon=True, name="ollama-warmup").start()


def _chunk_text(text: str, words: int = 6) -> Any:
    """Yield a text in small word chunks so the frontend can stream it
    smoothly even when the answer is produced instantly from the data."""
    tokens = text.split(" ")
    for index in range(0, len(tokens), words):
        piece = " ".join(tokens[index : index + words])
        yield piece + (" " if index + words < len(tokens) else "")


class RAGService:
    def __init__(self, data_path: str | os.PathLike[str]) -> None:
        self.data_path = Path(data_path)
        data_dir = self.data_path.parent
        # The models directory always lives alongside the data directory
        # (project_root/models/), regardless of how data_path is constructed.
        self.base_dir = data_dir.parent
        self.index_path = self.base_dir / "models" / "faiss.index"
        self.metadata_path = self.base_dir / "models" / "metadata.pkl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.loader = ExcelLoader(self.data_path)
        self.retriever = RetrievalService(self.data_path, self.index_path, self.metadata_path)
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            self.rebuild_index()

    def rebuild_index(self) -> dict[str, Any]:
        dataframe = self.loader.load()
        self.retriever.build_index(dataframe)
        return {"status": "ok", "records": int(len(dataframe))}

    def _build_generalized_summary(
        self,
        cause_query: str,
        semantic_results: list[dict[str, Any]],
        use_ai: bool = True,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Generate a generalized recommendation using AI (when use_ai is
        True), or fall back to the closest historical cases (keeping their
        REAL similarity scores — no fabricated numbers).

        Returns
        -------
        tuple[str, str, list[dict[str, Any]]]
            (recommended_solution, recommended_effort, similar_cases_context)
        """
        # --- Build context from the semantically similar cases ---
        context_cases = semantic_results[:5] if semantic_results else []

        # --- Attempt AI-powered generalized summary via Ollama (free, local) ---
        # Deadline-bounded and cached: never blocks the request on a cold model.
        if use_ai:
            answer = _ai_answer(cause_query, context_cases)
            if answer:
                return (
                    answer,
                    "Review the AI-generated guidance above and coordinate with the responsible team if the issue is unresolved.",
                    context_cases,
                )

        # --- Fallback: compose the recommendation from the closest cases ---
        best_cases = sorted(
            context_cases,
            key=lambda case: case.get("similarity", 0),
            reverse=True,
        )

        solution_options = [str(case.get("solution") or "").strip() for case in best_cases]
        solution_options = [option for option in solution_options if option]
        effort_options: list[str] = []
        for case in best_cases:
            for part in str(case.get("effort_taken") or "").split("|"):
                part = part.strip()
                if part and part not in effort_options:
                    effort_options.append(part)

        if solution_options:
            solution_summary = summarize_text(solution_options[0])
            if len(solution_options) > 1:
                related = " | ".join(summarize_text(option) for option in solution_options[1:3])
                solution_summary += f" Related resolutions from similar escalations: {related}"
        else:
            solution_summary = (
                "No directly matching escalation was found. Validate the issue against the "
                "closest historical cases and apply the standard resolution for this failure mode."
            )

        if effort_options:
            effort_summary = " ".join(
                f"{number}. {summarize_text(option)}"
                for number, option in enumerate(effort_options[:3], start=1)
            )
        else:
            effort_summary = (
                "Coordinate with the responsible department and escalate to the next "
                "level if the issue remains unresolved."
            )

        return solution_summary, effort_summary, best_cases

    # A search result is treated as a direct match when its combined
    # semantic + keyword score reaches this threshold.
    MATCH_THRESHOLD = 0.62

    def search(self, cause_query: str) -> dict[str, Any]:
        self.retriever.load_index() if self.index_path.exists() else self.rebuild_index()
        results = self.retriever.search(cause_query, top_k=5)

        # Clean the full text (no truncation) for accurate display.
        for item in results:
            item["cause"] = clean_text(item.get("cause", ""))
            item["solution"] = clean_text(item.get("solution", ""))
            item["effort_taken"] = clean_text(item.get("effort_taken", ""))

        top_result = next(
            (item for item in results if item["similarity"] >= self.MATCH_THRESHOLD),
            None,
        )

        if top_result is None:
            # No high-confidence direct match: use AI-powered generalized summary
            solution_summary, effort_summary, similar_cases = self._build_generalized_summary(
                cause_query=cause_query,
                semantic_results=results,
            )
            for case in similar_cases:
                case["cause"] = clean_text(case.get("cause", ""))
                case["solution"] = clean_text(case.get("solution", ""))
                case["effort_taken"] = clean_text(case.get("effort_taken", ""))
            first_case = similar_cases[0] if similar_cases else {}
            return {
                "recommended_solution": solution_summary.strip(),
                "recommended_effort": effort_summary.strip(),
                "escalation_id": first_case.get("escalation_id", ""),
                "timestamp": first_case.get("timestamp", ""),
                "department": first_case.get("department", ""),
                "customer_name": first_case.get("customer_name", ""),
                "material_name": first_case.get("material_name", ""),
                "kp_no": first_case.get("kp_no", ""),
                "level": first_case.get("level", ""),
                "status": first_case.get("status", ""),
                "match_type": "generalized",
                "similar_cases": similar_cases,
            }

        return {
            "recommended_solution": clean_text(top_result.get("solution", "")),
            "recommended_effort": clean_text(top_result.get("effort_taken", "")),
            "escalation_id": top_result.get("escalation_id", ""),
            "timestamp": top_result.get("timestamp", ""),
            "department": top_result.get("department", ""),
            "customer_name": top_result.get("customer_name", ""),
            "material_name": top_result.get("material_name", ""),
            "kp_no": top_result.get("kp_no", ""),
            "level": top_result.get("level", ""),
            "status": top_result.get("status", ""),
            "match_type": "exact",
            "similar_cases": results,
        }

    def stream_answer(self, cause_query: str, semantic_results: list[dict[str, Any]]) -> Any:
        """Yield the recommended answer in small text chunks.

        Fast path: when the hybrid search already found a high-confidence
        historical match, the stored solution/effort trail is returned
        instantly — no AI generation wait. The local Ollama model is only
        used for novel causes with no strong match.
        """
        top = semantic_results[0] if semantic_results else None
        if top and top.get("similarity", 0) >= self.MATCH_THRESHOLD:
            solution = clean_text(top.get("solution", "")) or (
                "No recorded solution for this escalation. Review the similar cases and apply the standard resolution for this failure mode."
            )
            effort = clean_text(top.get("effort_taken", "")) or (
                "Coordinate with the responsible department and escalate to the next level if the issue remains unresolved."
            )
            text = f"Recommended Solution:\n{solution}\n\nRecommended Effort/Action:\n{effort}"
            yield from _chunk_text(text)
            return

        # Slow path (novel cause): show a data-driven recommendation instantly,
        # then enhance it with the local AI, which streams in and replaces the
        # text once its first token arrives (frontend resets on "ai_start").
        #
        # The AI phase is deadline-bound (AI_FIRST_TOKEN_TIMEOUT /
        # AI_TOTAL_TIMEOUT) and cached per cause, so this request always
        # finishes quickly even when the model is cold or the machine is slow.
        solution_summary, effort_summary, _ = self._build_generalized_summary(
            cause_query,
            semantic_results,
            use_ai=False,
        )
        instant_text = (
            f"Recommended Solution:\n{solution_summary}\n\nRecommended Effort/Action:\n{effort_summary}"
        )
        yield from _chunk_text(instant_text)

        cache_key = cause_query.strip().lower()
        cached = _ai_cache_get(cache_key)
        if cached:
            # Same cause asked before — replay the stored answer instantly.
            yield {"type": "ai_start"}
            yield from _chunk_text(cached, words=12)
            yield {"type": "ai_end", "truncated": False, "cached": True}
            return

        context_cases = semantic_results[:AI_MAX_CONTEXT_CASES] if semantic_results else []
        completed: list = []
        pieces: list[str] = []
        for piece in _iter_ai_tokens(cause_query, context_cases, completed):
            if not pieces:
                yield {"type": "ai_start"}
            pieces.append(piece)
            yield piece

        if not pieces:
            # Model cold / Ollama unavailable / nothing inside the deadline:
            # the instant data-driven answer already sent stays on screen.
            yield {"type": "ai_skip", "reason": "deadline"}
            return

        answer = "".join(pieces).strip()
        if completed:
            _ai_cache_put(cache_key, answer)
        yield {"type": "ai_end", "truncated": not completed}

    def warmup(self) -> dict[str, Any]:
        """Load the embedding model and FAISS index before serving traffic.

        The first `sentence_transformers` encode costs ~30 s on a cold CPU-only
        host, and that cost used to be paid by the user's first search. Warming
        up at startup keeps every search sub-second; `/health` is served by a
        background thread so the server still answers immediately.
        """

        def _warm() -> None:
            try:
                if not self.index_path.exists() or not self.metadata_path.exists():
                    self.rebuild_index()
                else:
                    self.retriever.load_index()
                self.retriever.embedding_service.embed(["warmup"])
            except Exception:
                pass  # search() will retry the lazy load on demand

        thread = threading.Thread(target=_warm, daemon=True, name="embedder-warmup")
        thread.start()
        return {"status": "warming"}

    def save_new_record(self, cause: str, prevention: str, effort_taken: str, status: str) -> dict[str, Any]:
        """Append a new escalation to the source file (in its ORIGINAL schema,
        keeping all existing multi-level rows) and refresh the index so the
        new record is immediately searchable."""
        assigned_id = self.loader.append_record(
            {
                "cause": clean_text(cause),
                "prevention": clean_text(prevention),
                "solution": clean_text(prevention),
                "effort_taken": clean_text(effort_taken),
                "status": (status or "").strip(),
            }
        )
        self.rebuild_index()
        return {"status": "saved", "escalation_id": assigned_id}
