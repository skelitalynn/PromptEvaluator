from llm_helpers import call_llm_safe


class APITimeoutError(Exception):
    pass


class RateLimitError(Exception):
    pass


class FlakyLLM:
    def __init__(self):
        self.calls = 0

    def think(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise APITimeoutError("timeout")
        return "ok after retry"


class EmptyThenSuccessLLM:
    def __init__(self):
        self.calls = 0

    def think(self, messages):
        self.calls += 1
        if self.calls == 1:
            return ""
        return "filled"


class AlwaysRateLimitedLLM:
    def think(self, messages):
        raise RateLimitError("rate limited")


def test_call_llm_safe_retries_on_timeout():
    llm = FlakyLLM()
    result = call_llm_safe(
        llm,
        [{"role": "user", "content": "hello"}],
        max_retries=2,
        base_backoff_seconds=0,
    )
    assert result["ok"] is True
    assert result["content"] == "ok after retry"
    assert result["attempts"] == 2


def test_call_llm_safe_retries_on_empty_response():
    llm = EmptyThenSuccessLLM()
    result = call_llm_safe(
        llm,
        [{"role": "user", "content": "hello"}],
        max_retries=2,
        base_backoff_seconds=0,
    )
    assert result["ok"] is True
    assert result["content"] == "filled"
    assert result["attempts"] == 2


def test_call_llm_safe_returns_typed_error_on_failure():
    llm = AlwaysRateLimitedLLM()
    result = call_llm_safe(
        llm,
        [{"role": "user", "content": "hello"}],
        max_retries=2,
        base_backoff_seconds=0,
    )
    assert result["ok"] is False
    assert result["error_type"] == "rate_limit"
    assert result["attempts"] == 3
