"""Tool wrapper for dense / graph retrieval used by the orchestrator."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import app.services as services
except ImportError:  # pragma: no cover - direct execution fallback
    import services


def retrieve(query: str, *, user_id: Optional[str] = None, strategy: str = "hybrid") -> Dict[str, Any]:
    result = services.query_graph_with_trace(query, user_id=user_id)
    trace = dict(result.get("trace") or {})
    trace["selector_strategy"] = strategy
    return {"documents": result.get("documents") or [], "trace": trace}
