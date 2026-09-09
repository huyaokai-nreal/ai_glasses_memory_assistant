"""Unit tests for the continuous e2e scorer: cpCER, fail-closed checks, replay boundaries.

No AliMeeting audio and no ASR model are needed: the cpCER and contract tests are
pure, and the service replay test uses synthetic ``audio_event.v1`` payloads in a
throwaway home.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import score_continuous_e2e as scorer  # noqa: E402
from ai_glasses_memory_assistant.audio_engine import AudioEvent  # noqa: E402
from ai_glasses_memory_assistant.turn_planner import plan_audio_event  # noqa: E402


# --------------------------------------------------------------------------- #
# cpCER
# --------------------------------------------------------------------------- #


def test_cpcer_is_permutation_invariant() -> None:
    reference = {"A": "今天天气不错", "B": "明天要去开会"}
    hypothesis = {"spk_01": "明天要去开会", "spk_02": "今天气不错"}
    result = scorer.concatenated_cer(reference, hypothesis)
    assert result["mapping"] == {"spk_01": "B", "spk_02": "A"}
    assert result["cer"] == 1 / 12


def test_cpcer_handles_missing_track_as_deletions() -> None:
    reference = {"A": "四个字的话啊", "B": "另外一个人"}
    hypothesis = {"spk_01": "四个字的话啊"}
    result = scorer.concatenated_cer(reference, hypothesis)
    assert result["hypothesis_tracks"] == 1
    assert result["reference_speakers"] == 2
    assert result["cer"] == 5 / 11


def test_cpcer_handles_extra_track_as_insertions() -> None:
    reference = {"A": "四个字的话啊"}
    hypothesis = {"spk_01": "四个字的话啊", "spk_02": "多余的一句话"}
    result = scorer.concatenated_cer(reference, hypothesis)
    assert result["cer"] == 6 / 6


def test_cpcer_handles_empty_tracks() -> None:
    reference = {"A": ""}
    hypothesis = {"spk_01": ""}
    result = scorer.concatenated_cer(reference, hypothesis)
    assert result["cer"] is None


def test_cpcer_scores_four_tracks() -> None:
    reference = {name: f"{name}的内容" for name in ("A", "B", "C", "D")}
    hypothesis = {"spk_01": "A的内容", "spk_02": "B的内容", "spk_03": "C的内容", "spk_04": "D的内容"}
    result = scorer.concatenated_cer(reference, hypothesis)
    assert result["cer"] == 0.0
    assert len(result["mapping"]) == 4


# --------------------------------------------------------------------------- #
# cpCER optimization equivalence (pre-computed pair costs vs. per-pair score_cer)
# --------------------------------------------------------------------------- #


def _legacy_concatenated_cer(
    reference_by_speaker: dict[str, str], hypothesis_by_track: dict[str, str]
) -> dict:
    """Exact copy of the pre-optimization ``concatenated_cer`` (per-pair score_cer)."""
    ref_keys = sorted(reference_by_speaker)
    hyp_keys = sorted(hypothesis_by_track)
    best = None
    if len(hyp_keys) <= len(ref_keys):
        combinations = [
            list(zip(hyp_keys, assignment)) for assignment in scorer.itertools.permutations(ref_keys, len(hyp_keys))
        ]
    else:
        combinations = [
            list(zip(assignment, ref_keys)) for assignment in scorer.itertools.permutations(hyp_keys, len(ref_keys))
        ]
    if not combinations:
        combinations = [[]]
    for pairs in combinations:
        errors = 0
        characters = 0
        used_ref = set()
        used_hyp = set()
        for track, speaker in pairs:
            normalized = scorer.score_cer(reference_by_speaker[speaker], hypothesis_by_track[track])["normalized"]
            errors += int(normalized["errors"])
            characters += int(normalized["reference_chars"])
            used_ref.add(speaker)
            used_hyp.add(track)
        for speaker in ref_keys:
            if speaker in used_ref:
                continue
            normalized = scorer.score_cer(reference_by_speaker[speaker], "")["normalized"]
            errors += int(normalized["errors"])
            characters += int(normalized["reference_chars"])
        for track in hyp_keys:
            if track in used_hyp:
                continue
            normalized = scorer.score_cer("", hypothesis_by_track[track])["normalized"]
            errors += int(normalized["errors"])
        candidate = {
            "errors": errors,
            "reference_chars": characters,
            "cer": (errors / characters) if characters else None,
            "mapping": {track: speaker for track, speaker in pairs},
        }
        if best is None or (candidate["cer"] is not None and (best["cer"] is None or candidate["cer"] < best["cer"])):
            best = candidate
    assert best is not None
    best["reference_speakers"] = len(ref_keys)
    best["hypothesis_tracks"] = len(hyp_keys)
    return best


def _assert_equivalent(reference: dict[str, str], hypothesis: dict[str, str]) -> dict:
    old = _legacy_concatenated_cer(reference, hypothesis)
    new = scorer.concatenated_cer(reference, hypothesis)
    assert old == new, f"optimization changed the result:\nold={old}\nnew={new}"
    return new


def test_cpcer_optimization_equivalent_four_tracks() -> None:
    reference = {name: f"{name}的内容" for name in ("A", "B", "C", "D")}
    hypothesis = {"spk_01": "A的内容", "spk_02": "B的内容", "spk_03": "C的内容", "spk_04": "D的内容"}
    result = _assert_equivalent(reference, hypothesis)
    assert result["errors"] == 0
    assert len(result["mapping"]) == 4


def test_cpcer_optimization_equivalent_missing_track() -> None:
    _assert_equivalent({"A": "四个字的话啊", "B": "另外一个人"}, {"spk_01": "四个字的话啊"})


def test_cpcer_optimization_equivalent_extra_track() -> None:
    _assert_equivalent({"A": "四个字的话啊"}, {"spk_01": "四个字的话啊", "spk_02": "多余的一句话"})


def test_cpcer_optimization_equivalent_empty_text() -> None:
    _assert_equivalent({"A": ""}, {"spk_01": ""})
    _assert_equivalent({"A": "有内容", "B": ""}, {"spk_01": "有内容"})


def test_cpcer_optimization_equivalent_normalization() -> None:
    # Full-width characters, punctuation and spaces must be normalized identically.
    _assert_equivalent({"A": "你好，世界！", "B": "ａｂｃ １２３"}, {"spk_01": "你好世界", "spk_02": "abc123"})


def test_cpcer_optimization_equivalent_tie_mapping() -> None:
    # Two identical speakers and two identical tracks: several permutations tie,
    # the first one (sorted order) must win in both implementations.
    reference = {"A": "abc", "B": "abc"}
    hypothesis = {"spk_01": "abc", "spk_02": "abc"}
    old = _legacy_concatenated_cer(reference, hypothesis)
    new = scorer.concatenated_cer(reference, hypothesis)
    assert old == new
    assert new["mapping"] == old["mapping"] == {"spk_01": "A", "spk_02": "B"}


def test_cpcer_normalizes_punctuation_and_width() -> None:
    # Punctuation and full-width forms disappear under normalize_text, so the
    # scored CER is identical to the already-normalized reference.
    punctuated = scorer.concatenated_cer({"A": "你好，世界！"}, {"spk_01": "你好世界"})
    clean = scorer.concatenated_cer({"A": "你好世界"}, {"spk_01": "你好世界"})
    assert punctuated["cer"] == clean["cer"] == 0.0


# --------------------------------------------------------------------------- #
# Fail-closed input checks
# --------------------------------------------------------------------------- #


def _row(identifier: str, start: float, end: float, variant: str = "all8", track: str = "spk_01") -> dict:
    return {"segment_id": identifier, "variant": variant, "track": track, "start_s": start, "end_s": end}


def test_segment_rows_reject_duplicate_ids() -> None:
    rows = [_row("a", 0.0, 1.0), _row("a", 1.0, 2.0)]
    assert scorer.check_segment_rows(rows, 10.0)["duplicate_segment_ids"] == ["a"]


def test_segment_rows_reject_duplicate_time_and_range() -> None:
    rows = [_row("a", 0.0, 2.0), _row("b", 1.0, 3.0), _row("c", 0.0, 20.0)]
    result = scorer.check_segment_rows(rows, 10.0)
    assert result["duplicate_track_intervals"]
    assert result["out_of_range_intervals"] == ["c"]
    assert result["passed"] is False


def test_verify_frontend_against_prep_detects_hash_drift() -> None:
    manifest = {
        "case": {"case_id": "C", "cut_wav_sha256": "aaa"},
        "model": {"sha256": scorer.SORTFORMER_MODEL_SHA256},
        "inputs": {
            "sortformer_run_manifest": "/nonexistent/run-manifest.json",
            "sortformer_run_manifest_sha256": "x",
            "probabilities_path": "/nonexistent/probs.npy",
            "probabilities_sha256": "y",
        },
        "integrity": {"passed": True},
    }
    case = {"case_id": "C", "cut_wav_sha256": "bbb"}
    result = scorer.verify_frontend_against_prep(manifest, case)
    assert not result["passed"]
    assert any("WAV hash drift" in problem for problem in result["problems"])


def test_verify_frontend_against_prep_detects_model_drift() -> None:
    manifest = {
        "case": {"case_id": "C", "cut_wav_sha256": "aaa"},
        "model": {"sha256": "not-the-pinned-model"},
        "inputs": {},
        "integrity": {"passed": True},
    }
    case = {"case_id": "C", "cut_wav_sha256": "aaa"}
    result = scorer.verify_frontend_against_prep(manifest, case)
    assert any("Sortformer model hash drift" in problem for problem in result["problems"])


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #


def test_slice_track_audio_concatenates_overlapping_pieces() -> None:
    rows = [
        {"segment_id": "a", "start_s": 0.0, "end_s": 1.0, "track": "spk_01"},
        {"segment_id": "b", "start_s": 1.0, "end_s": 2.0, "track": "spk_01"},
    ]
    audio = {
        "a": np.arange(0, 16_000, dtype=np.float32),
        "b": np.arange(16_000, 32_000, dtype=np.float32),
    }
    sliced = scorer.slice_track_audio(audio, rows, 0.5, 1.5)
    assert sliced.shape == (16_000,)
    assert np.allclose(sliced[:8_000], audio["a"][8_000:16_000])
    assert np.allclose(sliced[8_000:], audio["b"][:8_000])


def test_slice_track_audio_returns_silence_when_uncovered() -> None:
    rows = [{"segment_id": "a", "start_s": 0.0, "end_s": 1.0, "track": "spk_01"}]
    audio = {"a": np.zeros(16_000, dtype=np.float32)}
    # No overlap with [5,6)s => the fixed-timeline window is filled with silence,
    # but it must still be exactly round((6-5)*16000) samples long (P2-2).
    sliced = scorer.slice_track_audio(audio, rows, 5.0, 6.0)
    assert sliced.shape == (16_000,)
    assert np.all(sliced == 0.0)


# --------------------------------------------------------------------------- #
# Event contract and service boundaries
# --------------------------------------------------------------------------- #


def _segment_row(**overrides: object) -> dict:
    row = {
        "segment_id": "S__spk_01__000000000-000001000__all8",
        "session_id": "S",
        "variant": "all8",
        "track": "spk_01",
        "start_s": 0.0,
        "end_s": 1.0,
        "confidence": 0.9,
        "overlap": False,
        "fallback_reason": None,
    }
    row.update(overrides)  # type: ignore[arg-type]
    return row


def test_qualified_transcript_produces_anonymous_capture_event() -> None:
    event = scorer.build_audio_event(_segment_row(), "明天下午提交材料", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    assert event["schema_version"] == "audio_event.v1"
    assert event["type"] == "transcript_final"
    assert event["lane"] == "ambient"
    assert event["speaker"] == {
        "state": "unknown",
        "reason": "anonymous_diarization_track",
        "voice_group": "spk_01",
        "track_confidence": 0.9,
        "track_scope": "capture",
    }
    assert event["overlap"] == {"state": "not_observed"}
    assert "embedding" not in event and "pcm" not in event
    plan = plan_audio_event(AudioEvent.from_dict(event))
    assert plan.action == "capture"
    assert plan.memory_eligible is False


def test_overlap_marks_suspected_state() -> None:
    event = scorer.build_audio_event(_segment_row(overlap=True), "内容", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    assert event["overlap"] == {"state": "suspected"}


def test_empty_text_fallback_and_low_confidence_become_speech_rejected() -> None:
    rejected = scorer.build_audio_event(_segment_row(), "", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    assert rejected["type"] == "speech_rejected"
    assert rejected["text"] == ""
    assert rejected["vad"]["reason"] == "empty_asr_text"

    fallback = scorer.build_audio_event(
        _segment_row(fallback_reason="insufficient_target_solo_frames"),
        "有内容",
        variant="all8",
        asr_backend="sherpa_sensevoice",
        asr_sha="sha",
        threshold=0.30,
    )
    assert fallback["type"] == "speech_rejected"
    assert fallback["vad"]["reason"] == "track_enhancement_failed"

    low = scorer.build_audio_event(_segment_row(confidence=0.1), "有内容", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    assert low["type"] == "speech_rejected"
    assert low["vad"]["reason"] == "low_track_confidence"


def test_partial_events_stay_ui_only() -> None:
    payload = dict(
        scorer.build_audio_event(_segment_row(), "临时文本", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    )
    payload.update({"event_id": "partial-1", "type": "transcript_partial", "final": False})
    plan = plan_audio_event(AudioEvent.from_dict(payload))
    assert plan.action == "ui_only"
    assert plan.memory_eligible is False


def test_anonymous_final_enters_capture_without_memory_and_rejected_is_dropped() -> None:
    from ai_glasses_memory_assistant.agent_bridge import GlassesChatService
    from tests.helpers import isolated_app_home

    final = scorer.build_audio_event(_segment_row(), "明天下午提交材料", variant="all8", asr_backend="sherpa_sensevoice", asr_sha="sha", threshold=0.30)
    rejected = scorer.build_audio_event(
        _segment_row(segment_id="S__spk_02__000001000-000002000__all8", track="spk_02", start_s=1.0, end_s=2.0),
        "",
        variant="all8",
        asr_backend="sherpa_sensevoice",
        asr_sha="sha",
        threshold=0.30,
    )
    partial = dict(final)
    partial.update({"event_id": "partial-2", "type": "transcript_partial", "final": False})

    with _tmp_home() as home:
        with isolated_app_home(home):
            service = GlassesChatService()
            service.set_device_network_state(online=False)
            user_id = "u-e2e"
            capture = service.start_device_capture(user_id=user_id)
            before = len(service.memory_store.list_memories(user_id, limit=500))

            for payload in (final, rejected, partial):
                service.ingest_device_audio_event(
                    user_id=user_id, event_payload=payload, capture_id=capture["capture_id"]
                )
            service.wait_device_audio_event(user_id=user_id, event_id=final["event_id"], timeout=10.0)
            service.wait_device_audio_event(user_id=user_id, event_id=rejected["event_id"], timeout=10.0)

            persisted = service.timeline_store.get_capture(user_id, capture["capture_id"]) or {}
            chunks = persisted.get("chunks") or []
            assert [chunk["text"] for chunk in chunks] == ["明天下午提交材料"]
            assert len(service.memory_store.list_memories(user_id, limit=500)) == before
            service.timeline_store.finish_capture(
                user_id, capture["capture_id"], summary="interrupted", ended_at=0.0, status="interrupted"
            )
            service.close()


def _make_scores(
    *,
    cpcer_base: float = 0.50,
    cpcer_enh: float = 0.30,
    overlap_base: float = 0.40,
    overlap_enh: float = 0.25,
    qualified_finals: int = 3,
    der: float = 0.20,
    overlap_recall: float = 0.80,
    integrity_passed: bool = True,
):
    def variant(cer: float, ovl: float) -> dict:
        return {
            "integrity": {"passed": integrity_passed},
            "qualified_finals": qualified_finals,
            "cpcer": {"cer": cer},
            "intervals": {"overlap": {"normalized_cer": ovl}},
            "diarization": {"der": der, "overlap_frame_recall": overlap_recall},
        }

    return {"ch0": variant(cpcer_base, overlap_base), "all8": variant(cpcer_enh, overlap_enh)}


def _clean_replay() -> dict:
    return {
        "network_calls": 0,
        "memory_before": 0,
        "memory_after": 0,
        "capture_chunk_count": 3,
        "transcript_final_events": 3,
        "partial_events": 0,
        "ui_only_events": 0,
        "persisted_audio_event_ids": ["f1", "f2", "f3"],
    }


def test_gates_pass_when_all_conditions_met() -> None:
    scores = _make_scores()
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    # Without a real replay the acoustic gates can pass, but the full end-to-end
    # claim must NOT: "not executed" is never counted as "passed" (P1-2).
    assert gates["acoustic_passed"] is True
    assert gates["replay_evaluated"] is False
    assert gates["full_e2e_passed"] is False
    assert gates["passed"] is False
    assert all(gates["checks"][k] for k in (
        "duplicate_identities", "at_least_one_qualified_final", "cpcer_strictly_better",
        "overlap_cer_strictly_better", "rtf", "der_le_25", "overlap_recall_ge_70",
    ))


def test_gate_cpcer_strictly_better_fails_when_enhance_worse() -> None:
    scores = _make_scores(cpcer_base=0.30, cpcer_enh=0.50)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["cpcer_strictly_better"] is False
    assert gates["passed"] is False


def test_gate_overlap_cer_fails_when_enhance_worse() -> None:
    scores = _make_scores(overlap_base=0.25, overlap_enh=0.40)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["overlap_cer_strictly_better"] is False
    assert gates["passed"] is False


def test_gate_rtf_fails_when_over_one() -> None:
    scores = _make_scores()
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=1.2, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["rtf"] is False
    assert gates["passed"] is False


def test_gate_der_fails_when_over_25() -> None:
    scores = _make_scores(der=0.30)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["der_le_25"] is False
    assert gates["passed"] is False


def test_gate_overlap_recall_fails_when_below_70() -> None:
    scores = _make_scores(overlap_recall=0.60)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["overlap_recall_ge_70"] is False
    assert gates["passed"] is False


# --------------------------------------------------------------------------- #
# Gate scope: the same diarization numbers are a diagnostic in a 75s smoke run
# and an enforced failure in a full meeting run (brain-agent decision A).
# --------------------------------------------------------------------------- #


def _evaluate(scores: dict, *, scope: str, **kwargs) -> dict:
    return scorer.evaluate_gates(
        scores,
        baseline="ch0",
        enhance="all8",
        rtf=0.9,
        replay=_clean_replay(),
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
        gate_scope=scope,
        **kwargs,
    )


def test_low_overlap_recall_is_only_diagnostic_in_smoke_scope() -> None:
    """The retry2 75s value (0.5488) must not fail a smoke run."""
    scores = _make_scores(overlap_recall=0.5488)
    gates = _evaluate(scores, scope="smoke")
    assert gates["gate_scope"] == "smoke"
    assert gates["checks"]["overlap_recall_ge_70"] is False
    # Reported honestly, but explicitly not part of the pass decision.
    assert gates["diagnostics"]["overlap_frame_recall"]["value"] == 0.5488
    assert gates["diagnostics"]["overlap_frame_recall"]["meets"] is False
    assert gates["diagnostics"]["overlap_frame_recall"]["evaluable"] is True
    assert "overlap_recall_ge_70" not in gates["enforced_checks"]
    assert "overlap_recall_ge_70" in gates["diagnostic_checks"]
    assert gates["passed"] is True


def test_same_low_overlap_recall_fails_full_scope() -> None:
    """Identical numbers must fail once the run is judged at full scale."""
    scores = _make_scores(overlap_recall=0.5488)
    gates = _evaluate(scores, scope="full")
    assert gates["gate_scope"] == "full"
    assert gates["checks"]["overlap_recall_ge_70"] is False
    assert "overlap_recall_ge_70" in gates["enforced_checks"]
    assert gates["diagnostic_checks"] == []
    assert gates["diagnostics"] == {}
    assert gates["passed"] is False


def test_full_scope_fails_when_overlap_recall_missing() -> None:
    scores = _make_scores(overlap_recall=None)
    gates = _evaluate(scores, scope="full")
    # A missing metric is never treated as satisfied.
    assert gates["checks"]["overlap_recall_ge_70"] is False
    assert gates["passed"] is False


def test_full_scope_fails_when_der_missing() -> None:
    scores = _make_scores(der=None)
    gates = _evaluate(scores, scope="full")
    assert gates["checks"]["der_le_25"] is False
    assert gates["passed"] is False


def test_smoke_scope_reports_missing_metric_as_not_evaluable() -> None:
    scores = _make_scores(overlap_recall=None)
    gates = _evaluate(scores, scope="smoke")
    assert gates["diagnostics"]["overlap_frame_recall"]["evaluable"] is False
    assert gates["checks"]["overlap_recall_ge_70"] is False
    # Still not fatal in the smoke scope.
    assert gates["passed"] is True


def _full_meeting_manifest() -> dict:
    """A complete 1573.85 s meeting, no truncation."""
    return {
        "case": {
            "window_start_s": 0.0,
            "window_end_s": 1573.85,
            "source_duration_s": 1573.85,
            "processed_seconds": 1573.85,
            "limit_seconds": None,
        }
    }


def _smoke_manifest() -> dict:
    """A 75 s slice of the same meeting: limit set and processed < meeting."""
    manifest = _full_meeting_manifest()
    manifest["case"]["processed_seconds"] = 75.0
    manifest["case"]["limit_seconds"] = 75.0
    return manifest


def test_resolve_gate_scope_auto_follows_proven_truncation() -> None:
    # Truncated: positive limit and processed_seconds < meeting duration.
    assert scorer.resolve_gate_scope(_smoke_manifest()) == "smoke"
    # Not truncated: no limit, or processed == meeting length.
    assert scorer.resolve_gate_scope(_full_meeting_manifest()) == "full"
    # Unreadable or absent evidence falls back to the stricter scope so a full
    # meeting can never downgrade itself past the diarization gates.
    assert scorer.resolve_gate_scope({"case": {"limit_seconds": 75.0}}) == "full"
    assert scorer.resolve_gate_scope({"case": {}}) == "full"
    assert scorer.resolve_gate_scope({}) == "full"


def test_resolve_gate_scope_auto_rejects_unusable_metadata() -> None:
    # limit present but the comparison cannot be made => full.
    assert scorer.resolve_gate_scope({"case": {"limit_seconds": 75.0, "processed_seconds": 75.0}}) == "full"
    assert scorer.resolve_gate_scope(
        {"case": {"limit_seconds": 75.0, "processed_seconds": 75.0, "window_end_s": "x"}}
    ) == "full"
    assert scorer.resolve_gate_scope(
        {"case": {"limit_seconds": "abc", "processed_seconds": 75.0, "window_start_s": 0.0, "window_end_s": 100.0}}
    ) == "full"


def test_resolve_gate_scope_limit_longer_than_meeting_is_full() -> None:
    """--limit-seconds beyond the meeting length does not make it a smoke run."""
    manifest = _full_meeting_manifest()
    manifest["case"]["limit_seconds"] = 5000.0
    assert scorer.resolve_gate_scope(manifest) == "full"
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.resolve_gate_scope(manifest, "smoke")


def test_resolve_gate_scope_refuses_explicit_smoke_on_full_meeting() -> None:
    """A complete meeting must not be downgraded to the relaxed scope."""
    with pytest.raises(scorer.ContinuousScoreError, match="not provably truncated"):
        scorer.resolve_gate_scope(_full_meeting_manifest(), "smoke")


def test_resolve_gate_scope_full_override_strengthens() -> None:
    assert scorer.resolve_gate_scope(_smoke_manifest(), "full") == "full"
    assert scorer.resolve_gate_scope(_full_meeting_manifest(), "full") == "full"
    assert scorer.resolve_gate_scope({}, "full") == "full"


def test_resolve_gate_scope_smoke_override_ok_when_truncated() -> None:
    assert scorer.resolve_gate_scope(_smoke_manifest(), "smoke") == "smoke"


def test_resolve_gate_scope_override_and_validation() -> None:
    assert scorer.resolve_gate_scope(_smoke_manifest(), "full") == "full"
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.resolve_gate_scope(_smoke_manifest(), "nonsense")


def test_explain_gate_scope_reports_evidence() -> None:
    evidence = scorer.explain_gate_scope(_smoke_manifest())
    assert evidence["truncated"] is True
    assert evidence["meeting_seconds"] == pytest.approx(1573.85)
    assert evidence["processed_seconds"] == pytest.approx(75.0)
    assert evidence["reasons"] == []
    missing = scorer.explain_gate_scope({})
    assert missing["truncated"] is False
    assert missing["reasons"]


def test_evaluate_gates_rejects_unknown_scope() -> None:
    scores = _make_scores()
    with pytest.raises(scorer.ContinuousScoreError):
        _evaluate(scores, scope="nonsense")


def test_gate_at_least_one_qualified_final_fails_when_zero() -> None:
    scores = _make_scores(qualified_finals=0)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["at_least_one_qualified_final"] is False
    assert gates["passed"] is False


def test_gate_integrity_fails_when_variant_invalid() -> None:
    scores = _make_scores(integrity_passed=False)
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=None,
        diarization=scores["all8"]["diarization"],
    )
    assert gates["checks"]["duplicate_identities"] is False
    assert gates["passed"] is False


def test_replay_gates_fail_when_replay_violates() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    replay["network_calls"] = 1
    replay["memory_after"] = 2
    replay["capture_chunk_count"] = 5
    replay["partial_events"] = 1
    # Persisted set must equal the qualified-final IDs; a stray id fails that gate.
    replay["persisted_audio_event_ids"] = ["stray-id"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids=set(),
    )
    assert gates["replay_evaluated"] is True
    assert gates["checks"]["network_calls_zero"] is False
    assert gates["checks"]["memory_unchanged"] is False
    assert gates["checks"]["timeline_chunk_equals_finals"] is False
    assert gates["checks"]["partial_not_persisted"] is False
    assert gates["passed"] is False


def test_replay_gates_pass_when_persisted_ids_match_finals() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    replay["persisted_audio_event_ids"] = ["f1", "f2", "f3"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["checks"]["timeline_chunk_equals_finals"] is True
    assert gates["full_e2e_passed"] is True
    assert gates["passed"] is True


def test_replay_gates_fail_when_persisted_ids_drop_a_final() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    # A qualified final was dropped from the timeline.
    replay["persisted_audio_event_ids"] = ["f1", "f2"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["checks"]["timeline_chunk_equals_finals"] is False
    assert gates["passed"] is False


def test_replay_gates_pass_when_replay_clean() -> None:
    scores = _make_scores()
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=_clean_replay(),
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["replay_evaluated"] is True
    assert gates["checks"]["network_calls_zero"] is True
    assert gates["checks"]["memory_unchanged"] is True
    assert gates["checks"]["timeline_chunk_equals_finals"] is True
    assert gates["checks"]["partial_not_persisted"] is True
    assert gates["passed"] is True


def test_replay_gates_fail_when_persisted_has_none_or_empty() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    # A None / empty id is not a valid persisted final and must fail the gate (P3-2).
    replay["persisted_audio_event_ids"] = ["f1", None, "", "f3"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["checks"]["timeline_chunk_equals_finals"] is False
    assert gates["passed"] is False


def test_replay_gates_fail_when_persisted_has_duplicate() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    # A duplicated id with no extra distinct id means a final was double-counted.
    replay["persisted_audio_event_ids"] = ["f1", "f2", "f2"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["checks"]["timeline_chunk_equals_finals"] is False
    assert gates["passed"] is False


def test_replay_gates_fail_when_chunk_count_mismatch() -> None:
    scores = _make_scores()
    replay = dict(_clean_replay())
    # timeline chunk count must equal the replayed-final count and the expected id count.
    replay["capture_chunk_count"] = 4
    replay["persisted_audio_event_ids"] = ["f1", "f2", "f3", "f4"]
    gates = scorer.evaluate_gates(
        scores, baseline="ch0", enhance="all8", rtf=0.9, replay=replay,
        diarization=scores["all8"]["diarization"],
        expected_final_event_ids={"f1", "f2", "f3"},
    )
    assert gates["checks"]["timeline_chunk_equals_finals"] is False
    assert gates["passed"] is False


def _synthetic_final_event(seg_id: str, text: str, start: float, end: float, track: str = "spk_01") -> dict:
    row = {
        "segment_id": seg_id, "session_id": "S", "variant": "all8",
        "track": track, "start_s": start, "end_s": end,
        "confidence": 0.9, "overlap": False, "fallback_reason": None,
    }
    return scorer.build_audio_event(
        row, text, variant="all8", asr_backend="sherpa_sensevoice",
        asr_sha="deadbeef", threshold=0.30,
    )


def test_replay_events_persists_only_finals() -> None:
    """Real isolated-service replay (not a hand-built result dict).

    Verifies the persisted chunk metadata.audio_event_id set equals exactly the
    replayed final's real event id, and that injected rejected / partial events
    never enter the Timeline (P3-1 / P3-3).
    """
    final = _synthetic_final_event("S__spk_01__000000000-000001000__all8", "你好世界", 0.0, 1.0)
    rejected = _synthetic_final_event("S__spk_01__000001000-000002000__all8", "", 1.0, 2.0)  # empty -> speech_rejected
    partial = dict(final)
    partial["event_id"] = "partial-1"
    partial["type"] = "transcript_partial"
    partial["final"] = False
    partial["text"] = "片段"

    result = scorer.replay_events([final, rejected, partial], timeout=15.0)

    assert result["transcript_final_events"] == 1
    assert result["capture_chunk_count"] == 1
    # Exactly one chunk, keyed by the final's real event id inside metadata.
    assert result["persisted_audio_event_ids"] == [final["event_id"]]
    assert final["event_id"] in result["persisted_audio_event_ids"]
    # Rejected and partial must not leak into the timeline.
    assert rejected["event_id"] not in result["persisted_audio_event_ids"]
    assert "partial-1" not in result["persisted_audio_event_ids"]
    assert result["network_calls"] == 0


# --------------------------------------------------------------------------- #
# Timeline reconstruction: exact length, silence in holes, no overlap/gap
# --------------------------------------------------------------------------- #


def test_slice_track_audio_exact_sample_count_for_full_coverage() -> None:
    rows = [
        {"segment_id": "a", "start_s": 0.0, "end_s": 1.0, "track": "spk_01"},
        {"segment_id": "b", "start_s": 1.0, "end_s": 2.0, "track": "spk_01"},
    ]
    audio = {
        "a": np.arange(0, 16_000, dtype=np.float32),
        "b": np.arange(16_000, 32_000, dtype=np.float32),
    }
    sliced = scorer.slice_track_audio(audio, rows, 0.0, 2.0)
    assert sliced.shape == (32_000,)
    assert np.allclose(sliced[:16_000], audio["a"])
    assert np.allclose(sliced[16_000:], audio["b"])


def test_slice_track_audio_middle_hole_stays_silent() -> None:
    rows = [
        {"segment_id": "a", "start_s": 0.0, "end_s": 1.0, "track": "spk_01"},
        {"segment_id": "b", "start_s": 2.0, "end_s": 3.0, "track": "spk_01"},
    ]
    audio = {"a": np.full(16_000, 5.0, dtype=np.float32), "b": np.full(16_000, 9.0, dtype=np.float32)}
    sliced = scorer.slice_track_audio(audio, rows, 0.0, 3.0)
    assert sliced.shape == (48_000,)
    assert np.all(sliced[:16_000] == 5.0)
    assert np.all(sliced[16_000:32_000] == 0.0)  # hole [1,2)s stays silent
    assert np.all(sliced[32_000:] == 9.0)


def test_slice_track_audio_adjacent_segments_no_overlap_no_gap() -> None:
    rows = [
        {"segment_id": "a", "start_s": 0.0, "end_s": 1.0, "track": "spk_01"},
        {"segment_id": "b", "start_s": 1.0, "end_s": 2.0, "track": "spk_01"},
    ]
    audio = {"a": np.full(16_000, 3.0, dtype=np.float32), "b": np.full(16_000, 7.0, dtype=np.float32)}
    sliced = scorer.slice_track_audio(audio, rows, 0.0, 2.0)
    assert sliced.shape == (32_000,)
    assert np.all(sliced[:16_000] == 3.0)
    assert np.all(sliced[16_000:] == 7.0)


def test_slice_track_audio_partial_overlap_trims() -> None:
    rows = [{"segment_id": "a", "start_s": 0.5, "end_s": 1.5, "track": "spk_01"}]
    audio = {"a": np.full(16_000, 4.0, dtype=np.float32)}
    sliced = scorer.slice_track_audio(audio, rows, 0.0, 1.0)
    # Window [0,1)s intersects [0.5,1.5)s over [0.5,1.0)s => 0.5s = 8000 samples.
    assert sliced.shape == (16_000,)
    assert np.all(sliced[:8_000] == 0.0)
    assert np.all(sliced[8_000:] == 4.0)


# --------------------------------------------------------------------------- #
# Fail-closed hash checks
# --------------------------------------------------------------------------- #


def _write_wav(path: Path, samples: int, value: float = 0.0) -> None:
    sf.write(str(path), np.full(samples, value, dtype=np.float32), 16_000, subtype="PCM_16")


def test_load_segment_audio_requires_wav_sha256(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, 16_000)
    rows = [{"segment_id": "a", "wav_path": str(wav), "sample_count": 16_000}]
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.load_segment_audio(rows, tmp_path)


def test_load_segment_audio_rejects_wav_hash_drift(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, 16_000)
    rows = [{"segment_id": "a", "wav_path": str(wav), "wav_sha256": "deadbeef", "sample_count": 16_000}]
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.load_segment_audio(rows, tmp_path)


def test_load_segment_audio_rejects_sample_count_mismatch(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_wav(wav, 16_000)
    good = scorer.sha256_file(wav)
    rows = [{"segment_id": "a", "wav_path": str(wav), "wav_sha256": good, "sample_count": 15_999}]
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.load_segment_audio(rows, tmp_path)


def test_load_frontend_run_rejects_segments_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "segments.jsonl").write_text(
        json.dumps({"segment_id": "a", "wav_path": "a.wav", "sample_count": 16_000}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": scorer.FRONTEND_SCHEMA,
        "segments_jsonl_sha256": "wrong",
        "case": {"case_id": "C"},
        "integrity": {"passed": True},
    }
    (tmp_path / "frontend-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(scorer.ContinuousScoreError):
        scorer.load_frontend_run(tmp_path)


def test_load_frontend_run_accepts_matching_segments_hash(tmp_path: Path) -> None:
    seg_path = tmp_path / "segments.jsonl"
    seg_path.write_text(
        json.dumps({"segment_id": "a", "wav_path": "a.wav", "sample_count": 16_000}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": scorer.FRONTEND_SCHEMA,
        "segments_jsonl_sha256": scorer.sha256_file(seg_path),
        "case": {"case_id": "C"},
        "integrity": {"passed": True},
    }
    (tmp_path / "frontend-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded_manifest, rows = scorer.load_frontend_run(tmp_path)
    assert loaded_manifest["segments_jsonl_sha256"] == scorer.sha256_file(seg_path)
    assert len(rows) == 1


def test_write_report_emits_markdown(tmp_path: Path) -> None:
    payload = {
        "case_id": "C",
        "processed_seconds": 10.0,
        "asr_model_sha256": "x",
        "sortformer_model_sha256": "y",
        "full_loop_rtf": 0.5,
        "frontend_rtf": 0.1,
        "scoring_rtf": 0.2,
        "replay_rtf": 0.2,
        "variants": {
            "all8": {
                "cpcer": {"cer": 0.1},
                "intervals": {
                    "all": {"normalized_cer": 0.1},
                    "non_overlap": {"normalized_cer": 0.1},
                    "overlap": {"normalized_cer": 0.1},
                },
                "qualified_finals": 1,
                "rejected_events": 0,
                "fallback_segments": 0,
            }
        },
    }
    scorer.write_report(tmp_path / "REPORT.md", payload)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "连续多通道端到端评分报告" in text
    assert "all8" in text


def _tmp_home():
    import tempfile
    from contextlib import contextmanager

    @contextmanager
    def _manager():
        path = tempfile.mkdtemp(prefix="continuous-e2e-test-home-")
        try:
            yield path
        finally:
            pass

    return _manager()
