from __future__ import annotations

import json
from unittest.mock import patch

from ai_glasses_memory_assistant import import_helpers
from ai_glasses_memory_assistant import conversation_candidate_helpers
from ai_glasses_memory_assistant.evals import longmemeval_runner as runner


class _FakeSemanticAgent:
    """Captures run_conversation calls and returns scripted JSON responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def run_conversation(self, message, system_message=None, **kwargs):
        self.calls.append({"message": message, "system_message": system_message, "kwargs": kwargs})
        response = self._responses.pop(0) if self._responses else None
        if response is None:
            return {"final_response": "not json"}
        return {"final_response": json.dumps(response)}


def _batch_response(*entries):
    return {"results": list(entries)}


def test_batch_classification_maps_entries_by_index() -> None:
    agent = _FakeSemanticAgent([
        _batch_response(
            {"index": 0, "memory_action": "write", "kind": "profile", "memory_type": "preference", "confidence": 0.9, "reason": "r0"},
            {"index": 1, "memory_action": "none", "kind": "", "memory_type": "", "confidence": 0.1, "reason": "r1"},
            {"index": 2, "memory_action": "write", "kind": "event", "memory_type": "event", "confidence": 0.4, "reason": "low"},
        )
    ])
    decisions = import_helpers.classify_import_items_batch(
        ["I love coffee", "ok thanks", "bought a car"],
        semantic_agent=agent,
    )
    assert decisions is not None
    assert len(decisions) == 3
    assert decisions[0]["status"] == "accepted"
    assert decisions[0]["kind"] == "profile"
    assert decisions[1]["status"] == "rejected_low_confidence"
    assert decisions[2]["status"] == "rejected_low_confidence"  # confidence below threshold
    # The batch options must disable thinking and request JSON.
    assert agent.calls[0]["kwargs"]["disable_thinking"] is True
    assert agent.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_batch_classification_returns_none_on_invalid_json() -> None:
    agent = _FakeSemanticAgent([])  # returns "not json"
    assert import_helpers.classify_import_items_batch(["a"], semantic_agent=agent) is None


def test_batch_classification_returns_none_when_agent_raises() -> None:
    class _Raising:
        def run_conversation(self, message, system_message=None, **kwargs):
            raise RuntimeError("boom")

    assert import_helpers.classify_import_items_batch(["a"], semantic_agent=_Raising()) is None


def test_batch_classification_chunks_large_inputs() -> None:
    contents = [f"fact number {i}" for i in range(25)]
    agent = _FakeSemanticAgent([
        _batch_response(*[{"index": i, "memory_action": "write", "kind": "event", "memory_type": "event", "confidence": 0.9, "reason": "r"} for i in range(20)]),
        _batch_response(*[{"index": i, "memory_action": "write", "kind": "event", "memory_type": "event", "confidence": 0.9, "reason": "r"} for i in range(20, 25)]),
    ])
    decisions = import_helpers.classify_import_items_batch(contents, semantic_agent=agent)
    assert decisions is not None and len(decisions) == 25
    assert len(agent.calls) == 2  # 20 + 5


def test_batch_classification_disabled_via_env() -> None:
    with patch.dict("os.environ", {"AI_GLASSES_IMPORT_BATCH_CLASSIFY": "0"}):
        assert import_helpers.import_batch_classification_enabled() is False
    with patch.dict("os.environ", {}, clear=True):
        assert import_helpers.import_batch_classification_enabled() is True


def test_fragments_split_on_sentence_boundaries_only() -> None:
    fragments = conversation_candidate_helpers._conversation_fragments(
        "I like tea, coffee, and milk. I also enjoy hiking! What about you?"
    )
    # Commas inside a sentence no longer split; sentence-final punctuation does.
    assert fragments == [
        "I like tea, coffee, and milk",
        "I also enjoy hiking",
        "What about you",
    ]


def test_reader_thinking_extra_body_by_provider() -> None:
    assert runner._thinking_extra_body("llama_cpp", thinking="disabled") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert runner._thinking_extra_body("deepseek", thinking="disabled") == {
        "thinking": {"type": "disabled"}
    }
    assert runner._thinking_extra_body("llama_cpp", thinking="enabled") is None
    assert runner._thinking_extra_body("openai", thinking="disabled") is None


def test_judge_thinking_extra_body_llama_cpp() -> None:
    import importlib.util
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "judge_longmemeval", project_root / "scripts" / "judge_longmemeval.py"
    )
    judge_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_mod)

    with patch.dict("os.environ", {"AI_GLASSES_LLM_PROVIDER": "llama_cpp", "AI_GLASSES_JUDGE_THINKING": "disabled"}):
        assert judge_mod._judge_thinking_extra_body() == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    with patch.dict("os.environ", {"AI_GLASSES_LLM_PROVIDER": "llama_cpp", "AI_GLASSES_JUDGE_THINKING": "enabled"}):
        assert judge_mod._judge_thinking_extra_body() is None
