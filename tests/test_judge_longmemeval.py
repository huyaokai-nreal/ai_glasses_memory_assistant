import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "judge_longmemeval",
    PROJECT_ROOT / "scripts" / "judge_longmemeval.py",
)
judge_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(judge_mod)


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raw = self.responses.pop(0) if self.responses else ""
        message = type("M", (), {"content": raw})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions(responses)})()


def test_judge_retries_empty_response_then_succeeds() -> None:
    client = _FakeClient(["", "", '{"score": 1, "reason": "aligned"}'])

    result = judge_mod.judge_single(
        client,
        "fake-judge",
        "question",
        "reference",
        "hypothesis",
        retries=3,
    )

    assert result["score"] == 1
    assert result["reason"] == "aligned"
    assert client.chat.completions.calls == 3


def test_judge_returns_parse_error_after_all_failures() -> None:
    client = _FakeClient(["", ""])

    result = judge_mod.judge_single(
        client,
        "fake-judge",
        "question",
        "reference",
        "hypothesis",
        retries=2,
    )

    assert result == {"score": 0, "reason": "judge parse error"}
    assert client.chat.completions.calls == 2


def test_parse_json_robust_extracts_embedded_json() -> None:
    parsed = judge_mod._parse_json_robust('prefix {"score": 1, "reason": "good"} suffix')

    assert parsed is not None
    assert parsed["score"] == 1


def test_parse_json_robust_repairs_raw_newline_in_string() -> None:
    parsed = judge_mod._parse_json_robust('{"score": 1, "reason": "line1\nline2"}')

    assert parsed is not None
    assert parsed["score"] == 1
    assert "line1 line2" in parsed["reason"]
