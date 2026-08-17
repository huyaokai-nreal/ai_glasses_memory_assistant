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
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        raw = self.responses.pop(0) if self.responses else ""
        message = type("M", (), {"content": raw})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions(responses)})()


def test_judge_retries_empty_response_then_succeeds() -> None:
    client = _FakeClient(["", "", "yes"])

    result = judge_mod.judge_single(
        client,
        "fake-judge",
        "multi-session",
        "question",
        "reference",
        "hypothesis",
        retries=3,
    )

    assert result["score"] == 1
    assert result["reason"] == "judge response: yes"
    assert client.chat.completions.calls == 3


def test_judge_returns_parse_error_after_all_failures() -> None:
    client = _FakeClient(["", ""])

    result = judge_mod.judge_single(
        client,
        "fake-judge",
        "single-session-user",
        "question",
        "reference",
        "hypothesis",
        retries=2,
    )

    assert result == {"score": 0, "reason": "judge parse error"}
    assert client.chat.completions.calls == 2


def test_official_temporal_prompt_allows_off_by_one_counts() -> None:
    prompt = judge_mod.get_official_anscheck_prompt(
        "temporal-reasoning", "question", "18 days", "19 days"
    )

    assert "do not penalize off-by-one errors" in prompt
    assert "Answer yes or no only" in prompt


def test_official_preference_prompt_uses_rubric_contract() -> None:
    prompt = judge_mod.get_official_anscheck_prompt(
        "single-session-preference", "question", "rubric", "hypothesis"
    )

    assert "does not need to reflect all the points in the rubric" in prompt
    assert "recalls and utilizes the user's personal information correctly" in prompt


def test_official_abstention_prompt_accepts_identifying_unanswerable() -> None:
    prompt = judge_mod.get_official_anscheck_prompt(
        "multi-session",
        "question",
        "explanation",
        "I don't know based on the available memory.",
        abstention=True,
    )

    assert "unanswerable question" in prompt
    assert "information is incomplete" in prompt


def test_official_judge_uses_user_only_prompt_and_ten_tokens() -> None:
    client = _FakeClient(["no"])

    result = judge_mod.judge_single(
        client,
        "fake-judge",
        "knowledge-update",
        "question",
        "reference",
        "hypothesis",
    )

    assert result["score"] == 0
    assert result["reason"] == "judge response: no"
    assert client.chat.completions.calls == 1
    assert client.chat.completions.last_kwargs["max_tokens"] == 10
    assert client.chat.completions.last_kwargs["messages"][0]["role"] == "user"
    assert len(client.chat.completions.last_kwargs["messages"]) == 1
