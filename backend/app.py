import os
from contextlib import asynccontextmanager

# The project root `.env` must be loaded BEFORE the service modules are
# imported: `services.rag_service` reads its AI tuning knobs (model, token cap,
# deadlines, cache size) from the environment at import time.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_backend_dir, ".."))

from dotenv import load_dotenv  # noqa: E402  (must run before importing services)

load_dotenv(os.path.join(project_root, ".env"))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.routes import router, service # noqa: E402
from backend.services.rag_service import warm_ollama_async  # noqa: E402


def _cors_origins() -> list[str]:
    """Origins allowed to call the API (Vercel frontend + local dev).

    Set FRONTEND_URL on Render to the Vercel URL, e.g.
    https://escalation-app.vercel.app. Comma-separated values are supported.
    """
    raw = os.getenv("FRONTEND_URL", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    origins += ["http://127.0.0.1:3000", "http://localhost:3000"]
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(origins))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the embedding model + FAISS index now instead of on the user's first
    # search (that load costs ~30 s on a cold CPU-only host; the warmup holds
    # the load lock, so a concurrent search waits for it rather than building a
    # second copy of the model).
    service.warmup()
    # Loads the local model AND runs a realistic prompt through it, so the
    # first search does not pay the (multi-second) weight page-in.
    warm_ollama_async()
    yield


app = FastAPI(title="Escalation Management RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
