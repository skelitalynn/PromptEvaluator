class FakeLLM:
    """Simple deterministic LLM stub for tests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def think(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("FakeLLM has no more queued responses.")
        return self._responses.pop(0)
