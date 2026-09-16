import os
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.rag_service import RAGService

router = APIRouter()
base_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(base_dir, ".."))
excel_candidate = os.path.join(project_root, "data", "escalation.xlsx")
csv_candidate = os.path.join(project_root, "data", "Escalation Matrix - Escalation_Log.csv")
if os.path.exists(excel_candidate):
    data_path = excel_candidate
elif os.path.exists(csv_candidate):
    data_path = csv_candidate
else:
    raise FileNotFoundError(f"No data file found in {project_root}")
service = RAGService(data_path=data_path)


class SearchRequest(BaseModel):
    cause: str


class SaveRequest(BaseModel):
    cause: str
    prevention: str
    effort_taken: str
    status: str


@router.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    return service.search(request.cause)


@router.post("/search/stream")
def search_stream(request: SearchRequest) -> StreamingResponse:
    """Stream the AI solution from Ollama token-by-token (NDJSON lines).

    The first SSE event carries the search metadata + similar cases, and each
    following event carries a chunk of the AI-generated solution text.
    """
    import json

    def event_generator() -> AsyncGenerator[str, None]:
        # First, run semantic search to build context and to collect the
        # metadata + similar cases for the frontend.
        results = service.retriever.search(request.cause, top_k=5)
        first_case: Dict[str, Any] = results[0] if results else {}
        strong_match = bool(results) and results[0]["similarity"] >= service.MATCH_THRESHOLD
        meta = {
            "type": "meta",
            "escalation_id": first_case.get("escalation_id", ""),
            "timestamp": first_case.get("timestamp", ""),
            "department": first_case.get("department", ""),
            "customer_name": first_case.get("customer_name", ""),
            "material_name": first_case.get("material_name", ""),
            "kp_no": first_case.get("kp_no", ""),
            "level": first_case.get("level", ""),
            "status": first_case.get("status", ""),
            "match_type": "exact" if strong_match else "generalized",
            "similarity": results[0]["similarity"] if results else 0,
            "similar_cases": results,
        }
        yield f"data: {json.dumps(meta)}" + "\n\n"

        for chunk in service.stream_answer(request.cause, semantic_results=results):
            if isinstance(chunk, dict):
                yield f"data: {json.dumps(chunk)}" + "\n\n"
            else:
                payload = json.dumps({"type": "text", "content": chunk})
                yield f"data: {payload}" + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/save")
def save(request: SaveRequest) -> dict[str, Any]:
    try:
        return service.save_new_record(request.cause, request.prevention, request.effort_taken, request.status)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rebuild-index")
def rebuild_index() -> dict[str, Any]:
    return service.rebuild_index()
