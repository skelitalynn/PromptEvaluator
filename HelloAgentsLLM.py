import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class HelloAgentsLLM:
    """Lightweight LLM client wrapper with retry and structured error result."""

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        verbose: bool = True,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = apiKey or os.getenv("LLM_API_KEY")
        base_url = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.verbose = verbose

        if not all([self.model, api_key, base_url]):
            raise ValueError(
                "LLM_MODEL_ID, LLM_API_KEY, and LLM_BASE_URL must be provided via args or .env."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @staticmethod
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

    @staticmethod
    def _is_retryable(error_type: str) -> bool:
        return error_type in {
            "timeout",
            "rate_limit",
            "connection_error",
            "server_error",
            "empty_response",
            "unknown_error",
        }

    def think_result(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_retries: int = 2,
        base_backoff_seconds: float = 0.5,
        stream: bool = True,
    ) -> Dict:
        """Return a unified result payload for robust downstream handling."""
        last_error = {
            "ok": False,
            "content": "",
            "error_type": "unknown_error",
            "error_message": "Unknown failure.",
            "attempts": 0,
        }

        for attempt in range(1, max_retries + 2):
            if self.verbose:
                print(f"Calling model {self.model} (attempt {attempt}/{max_retries + 1})...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                )

                if stream:
                    collected_content = []
                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        if self.verbose:
                            print(content, end="", flush=True)
                        collected_content.append(content)
                    if self.verbose:
                        print()
                    final_content = "".join(collected_content).strip()
                else:
                    final_content = (response.choices[0].message.content or "").strip()

                if final_content:
                    return {
                        "ok": True,
                        "content": final_content,
                        "error_type": None,
                        "error_message": "",
                        "attempts": attempt,
                    }

                last_error = {
                    "ok": False,
                    "content": "",
                    "error_type": "empty_response",
                    "error_message": "LLM returned empty content.",
                    "attempts": attempt,
                }
            except Exception as exc:  # pragma: no cover - depends on provider/runtime
                error_type = self._classify_exception(exc)
                last_error = {
                    "ok": False,
                    "content": "",
                    "error_type": error_type,
                    "error_message": str(exc),
                    "attempts": attempt,
                }

            if attempt <= max_retries and self._is_retryable(last_error["error_type"]):
                sleep_seconds = base_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

        return last_error

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """Backward-compatible interface: return text only."""
        result = self.think_result(messages=messages, temperature=temperature)
        return result["content"] if result["ok"] else ""


if __name__ == "__main__":
    try:
        llm_client = HelloAgentsLLM()
        example_messages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a quicksort function in Python."},
        ]
        print("--- LLM Call ---")
        response_text = llm_client.think(example_messages)
        if response_text:
            print("\n--- Full Response ---")
            print(response_text)
    except ValueError as exc:
        print(exc)
