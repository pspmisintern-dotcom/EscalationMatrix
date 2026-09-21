import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))

# Load local .env when running locally.
# On Render, environment variables configured in the dashboard
# are still available through os.environ.
load_dotenv(os.path.join(_project_root, ".env"))


# ---------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from routes import router, service  # noqa: E402


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

def _cors_origins() -> list[str]:
    """
    Origins allowed to call this API.

    FRONTEND_URL can contain one or more comma-separated URLs.

    Example on Render:

        FRONTEND_URL=https://escalation-matrix-v52l.vercel.app
    """

    raw = os.getenv("FRONTEND_URL", "")

    origins = [
        origin.strip().rstrip("/")
        for origin in raw.split(",")
        if origin.strip()
    ]

    # Production frontend
    origins.append(
        "https://escalation-matrix-v52l.vercel.app"
    )

    # Local development
    origins.extend(
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # Remove duplicates while preserving order
    return list(dict.fromkeys(origins))


# ---------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Keep application startup fast.

    IMPORTANT:
    Do NOT perform the expensive RAG/Ollama warmup here.

    Render needs FastAPI to become healthy quickly. Running
    service.warmup() during startup can cause the service to
    remain unavailable or fail its health checks.
    """

    print("Starting Escalation Management API...")
    print("CORS origins:", _cors_origins())

    yield

    print("Shutting down Escalation Management API...")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Escalation Management RAG",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """
    Lightweight health check.

    This must remain fast and must not load the RAG model,
    FAISS index, Ollama, or external services.
    """

    return {"status": "ok"}
