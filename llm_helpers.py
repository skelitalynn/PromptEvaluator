import time
from typing import Any, Dict, List


def _classify_exception(exc: Exception) -> str:
    name = exc.__class__.__name__
    if name == "APITimeoutError":
        return "timeout"
    if name == "RateLimitError":
        return "rate_limit"
    if name == "APIConnectionError":
        return "connection_error"
    if name == "AuthenticationError":
        return "auth_error"
    if name == "BadRequestError":
        return "bad_request"
    if name == "PermissionDeniedError":
        return "permission_denied"
    if name == "NotFoundError":
        return "not_found"
    if name == "UnprocessableEntityError":
        return "unprocessable_entity"
    if name == "APIStatusError":
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return "server_error"
        return "api_status_error"
    return "unknown_error"


def _is_retryable(error_type: str) -> bool:
    return error_type in {
        "timeout",
        "rate_limit",
        "connection_error",
        "server_error",
        "empty_response",
        "unknown_error",
    }


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(result.get("ok", False)),
        "content": result.get("content", "") or "",
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message", ""),
        "attempts": int(result.get("attempts", 1)),
    }


def call_llm_safe(
    llm: Any,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    max_retries: int = 2,
    base_backoff_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Unified safe LLM call with retry and error classification."""
    if hasattr(llm, "think_result") and callable(getattr(llm, "think_result")):
        result = llm.think_result(
            messages=messages,
            temperature=temperature,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
        )
        return _normalize_result(result)

    last_error = {
        "ok": False,
        "content": "",
        "error_type": "unknown_error",
        "error_message": "Unknown failure",
        "attempts": 0,
    }

    for attempt in range(1, max_retries + 2):
        try:
            content = llm.think(messages) or ""
            if content.strip():
                return {
                    "ok": True,
                    "content": content,
                    "error_type": None,
                    "error_message": "",
                    "attempts": attempt,
                }
            last_error = {
                "ok": False,
                "content": "",
                "error_type": "empty_response",
                "error_message": "LLM returned empty response.",
                "attempts": attempt,
            }
        except Exception as exc:  # pragma: no cover - depends on runtime/provider
            error_type = _classify_exception(exc)
            last_error = {
                "ok": False,
                "content": "",
                "error_type": error_type,
                "error_message": str(exc),
                "attempts": attempt,
            }

        if attempt <= max_retries and _is_retryable(last_error["error_type"]):
            sleep_seconds = base_backoff_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    return last_error
