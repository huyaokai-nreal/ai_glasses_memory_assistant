"""Generate minimal, desensitized, committable fixtures for full8 batch tests.

Run once:  python tests/fixtures/full8/_gen_fixtures.py
Produces deterministic JSON with no absolute paths and no real transcripts.

The model sha256 values are the REAL locked constants (mirrored from the scorer /
driver) because round 2 validation compares a session's recorded hashes against
the fixed locks — a fixture that self-declares a placeholder would be an
evidence failure. The replay payload mirrors the real ``scores.service_replay``
block so the privacy / interrupted / chunk-integrity evidence can be verified
for real. Numbers are illustrative, not measured from real audio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO_FRONTEND = HERE.parents[2] / "evals" / "audio_frontend"
if str(AUDIO_FRONTEND) not in sys.path:
    sys.path.insert(0, str(AUDIO_FRONTEND))

import run_continuous_full8 as driver  # noqa: E402  (locked hashes + case ids)

# Alias kept for backward compatibility with existing tests that referenced the
# placeholder names; the VALUES are now the real locked hashes.
PLACEHOLDER_ASR = driver.LOCKED_ASR_MODEL_SHA256
PLACEHOLDER_SF = driver.LOCKED_SORTFORMER_MODEL_SHA256

CASES = list(driver.LOCKED_CASE_IDS)


def build_replay(*, interrupted=True, audit_embedding=False, audit_pcm=False,
                 chunk_from_rejected=False, partial=0, ui_only=0,
                 finals=None, chunks=None, ids=None, dedupe_ids=False):
    """A realistic ``service_replay`` evidence block (mirrors the scorer fields)."""
    n = 8 if finals is None else finals
    chunks = n if chunks is None else chunks
    ids = ([f"evt_{i}" for i in range(n)] if ids is None else list(ids))
    if dedupe_ids and ids:
        ids = ids + [ids[0]]  # make one duplicate
    replay = {
        "events_replayed": n + 1,
        "network_attempts": [],
        "network_calls": 0,
        "transcript_final_events": n,
        "speech_rejected_events": 1,
        "partial_events": partial,
        "ui_only_events": ui_only,
        "capture_chunk_count": chunks,
        "timeline_entries_added": 0,
        "memory_before": 0,
        "memory_after": 0,
        "capture_status": "running",
        "capture_status_after_finish": "interrupted" if interrupted else "stopped",
        "audit_bytes": 4096,
        "audit_contains_embedding": audit_embedding,
        "audit_contains_pcm": audit_pcm,
        "chunk_texts_are_from_rejected": chunk_from_rejected,
        "persisted_audio_event_ids": ids,
        "isolated_home": "/tmp/fixture-home",
    }
    return replay


def _variant(cpcer, ov, nov):
    """Build one variant block. ``cer``/``normalized_cer`` are DERIVED from the
    numerator/denominator (the scorer writes rate == errors/reference; the round-2
    numeric-consistency check would reject a self-consistent-but-mismatched fake).
    """
    e, r = cpcer
    ov_e, ov_r = ov
    nov_e, nov_r = nov
    return {
        "cpcer": {"errors": e, "reference_chars": r, "cer": (e / r) if r else None},
        "intervals": {
            "overlap": {"errors": ov_e, "reference_chars": ov_r,
                        "normalized_cer": (ov_e / ov_r) if ov_r else None},
            "non_overlap": {"errors": nov_e, "reference_chars": nov_r,
                            "normalized_cer": (nov_e / nov_r) if nov_r else None},
            "all": {"errors": ov_e + nov_e, "reference_chars": ov_r + nov_r,
                    "normalized_cer": None if (ov_r + nov_r) == 0 else (ov_e + nov_e) / (ov_r + nov_r)},
        },
    }


def _diarization(errors, ref_spk, det, ov_ref, der=None, recall=None):
    """Diarization block with a SELF-CONSISTENT der/recall.

    The numeric-consistency check compares rate against errors/reference with
    1e-9 tolerance, so any stored der/recall must equal the frame ratio. When the
    caller passes rates that already equal the ratio they are kept as-is;
    otherwise (e.g. a pre-rounded 0.1846 for 4727/25607) the value is snapped to
    the exact frame-derived ratio so fixtures never trip the consistency check.
    """
    if der is None or (ref_spk and abs(float(der) - errors / ref_spk) > 1e-9):
        der = (errors / ref_spk) if ref_spk else der
    if recall is None or (ov_ref and abs(float(recall) - det / ov_ref) > 1e-9):
        recall = (det / ov_ref) if ov_ref else recall
    return {
        "errors": errors, "reference_speaker_frames": ref_spk,
        "overlap_detected_frames": det, "overlap_reference_frames": ov_ref,
        "der": der, "overlap_frame_recall": recall,
        "reference_speakers": ["spk_a"], "hypothesis_speakers": ["spk_a"],
        "speaker_mapping": {"spk_a": "spk_a"},
    }


def build_scores(*, all8, ch0, dia, checks_overrides=None, replay_evaluated=True,
                 seg_ok=True, drift_ok=True, case_id="FIXTURE", service_replay=None,
                 gates_scope=None):
    checks = {
        "duplicate_identities": True,
        "at_least_one_qualified_final": True,
        "cpcer_strictly_better": all8["cpcer"]["cer"] < ch0["cpcer"]["cer"],
        "overlap_cer_strictly_better": all8["intervals"]["overlap"]["normalized_cer"]
        < ch0["intervals"]["overlap"]["normalized_cer"],
        "rtf": True,
        "network_calls_zero": True,
        "memory_unchanged": True,
        "timeline_chunk_equals_finals": True,
        "partial_not_persisted": True,
        # DER/recall stay in the scorer output (diagnostic on a real run) but the
        # full8 summarizer never labels a session from them: aggregate tier only.
        "der_le_25": dia["der"] <= 0.25,
        "overlap_recall_ge_70": dia["overlap_frame_recall"] >= 0.70,
    }
    if checks_overrides:
        checks.update(checks_overrides)
    all8_full = dict(all8)
    all8_full["diarization"] = dia
    ch0_full = dict(ch0)
    ch0_full["diarization"] = dia
    replay = service_replay if service_replay is not None else build_replay()
    scores = {
        "schema": "continuous_e2e_scores.v1",
        "case_id": case_id,
        "processed_seconds": 100.0,
        "gate_scope": gates_scope or "full",
        "asr_model_sha256": PLACEHOLDER_ASR,
        "sortformer_model_sha256": PLACEHOLDER_SF,
        # In-process (internal) RTF, rounded to 6 dp, equals the sum of the three
        # stage RTFs. The summarizer re-derives it (frontend+scoring+replay) and
        # verifies the scorer's rtf gate flag is truthful against it.
        "frontend_rtf": 0.10,
        "scoring_rtf": 0.20,
        "replay_rtf": 0.05,
        "full_loop_rtf": 0.35,
        "input_drift": {"problems": [], "passed": drift_ok},
        "segment_integrity": {"total_rows": 1, "passed": seg_ok},
        "variants": {"all8": all8_full, "ch0": ch0_full},
        "gates": {
            "passed": all(v for v in checks.values()),
            "gate_scope": gates_scope or "full",
            "replay_evaluated": replay_evaluated,
            "enforced_checks": list(checks.keys()),
            "diagnostic_checks": [],
            "checks": checks,
        },
        "service_replay": replay,
    }
    return scores


def timing(passed=True, rtf=0.35):
    return {
        "schema": "continuous_full_loop_timing.v1",
        "full_process_rtf": rtf,
        "full_process_rtf_passed": passed,
        "full_process_seconds": 35.0,
        "processed_seconds": 100.0,
        "frontend_exit_code": 0,
        "scorer_exit_code": 0,
    }


def write_session(name, scores, timing_obj):
    d = HERE / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    (d / "full-loop-timing.json").write_text(json.dumps(timing_obj, indent=2) + "\n", encoding="utf-8")


def main():
    # 1) PASS: all8<ch0 cpCER, overlap, all gates true, full replay evidence green
    all8 = _variant((245, 1000), (300, 2000), (200, 9000))
    ch0 = _variant((438, 1000), (400, 2000), (300, 9000))
    dia = _diarization(4727, 25607, 4103, 5697, 0.1846, 0.7202)
    write_session("session_pass", build_scores(all8=all8, ch0=ch0, dia=dia, case_id=CASES[0]), timing())

    # 2) QUALITY_FAIL: evidence all ok, replay ok; cpcer_strictly_better false.
    #    DER is NOT a per-session failure (aggregate-only) — session-level fail is
    #    cpcer regression, which keeps the attempt selected but makes the batch red.
    all8q = _variant((500, 1000), (300, 2000), (200, 9000))
    ch0q = _variant((438, 1000), (400, 2000), (300, 9000))
    diaq = _diarization(4727, 25607, 4103, 5697, 0.1846, 0.7202)
    qchecks = {"cpcer_strictly_better": False}
    write_session("session_quality_fail",
                  build_scores(all8=all8q, ch0=ch0q, dia=diaq, checks_overrides=qchecks,
                               case_id=CASES[0]),
                  timing())

    # 3) EVIDENCE_FAIL: network_calls_zero false (rest ok)
    efail = {"network_calls_zero": False}
    write_session("session_evidence_fail",
                  build_scores(all8=all8, ch0=ch0, dia=dia, checks_overrides=efail,
                               case_id=CASES[0]),
                  timing())

    # 4) NOT_EVALUABLE: all8 non_overlap reference_chars == 0 (legal zero-denominator)
    all8_ne = _variant((245, 1000), (300, 2000), (0, 0))
    write_session("session_not_evaluable",
                  build_scores(all8=all8_ne, ch0=ch0, dia=dia, case_id=CASES[0]),
                  timing())

    # 5) PRIVACY_FAIL — audit leaked a speaker embedding: raw replay evidence fails
    write_session("session_privacy_embedding",
                  build_scores(all8=all8, ch0=ch0, dia=dia, case_id=CASES[0],
                               service_replay=build_replay(audit_embedding=True)),
                  timing())

    # 6) PRIVACY_FAIL — capture was not interrupted (stop would schedule an LLM call)
    write_session("session_privacy_not_interrupted",
                  build_scores(all8=all8, ch0=ch0, dia=dia, case_id=CASES[0],
                               service_replay=build_replay(interrupted=False)),
                  timing())

    # 7) TIMING_MISSING — fixture WITHOUT full-loop-timing.json (evidence failure)
    d = HERE / "session_timing_missing"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(
        json.dumps(build_scores(all8=all8, ch0=ch0, dia=dia, case_id=CASES[0]), indent=2) + "\n",
        encoding="utf-8")

    # prep-manifest fixture (exactly the locked eight, locked order). Each case
    # carries window_start_s/window_end_s and the manifest a total_audio_seconds so
    # the driver can DERIVE the disk budget from the eight sessions' processed duration.
    # Real per-session processed durations (mirror
    # reports/p3_continuous_prep/20260907-full8/prep-manifest.json) so the derived
    # disk budget in fixture runs matches the real eight-session scale: 15138.215 s.
    windows_s = {
        "R8001_M8004-full": 1573.85,
        "R8003_M8001-full": 2068.0,
        "R8007_M8010-full": 1856.3219375,
        "R8007_M8011-full": 1861.546,
        "R8008_M8013-full": 2239.4705,
        "R8009_M8018-full": 1654.657,
        "R8009_M8019-full": 1973.926,
        "R8009_M8020-full": 1910.4435625,
    }
    prep = {
        "total_audio_seconds": round(sum(windows_s[c] for c in CASES), 3),
        "cases": [
            {"case_id": c, "cut_wav_sha256": f"wavsha_{i:02d}", "textgrid_sha256": f"tgsha_{i:02d}",
             "window_start_s": 0.0, "window_end_s": windows_s[c]}
            for i, c in enumerate(CASES)
        ],
    }
    (HERE / "prep-manifest.json").write_text(json.dumps(prep, indent=2) + "\n", encoding="utf-8")

    # source-run manifest: the driver reads the ASR hash it will enforce from
    # runtime.ambient_audio_profile.asr_model_sha256 (same path as the real reports).
    src = HERE / "source-run"
    src.mkdir(parents=True, exist_ok=True)
    (src / "run-manifest.json").write_text(json.dumps({
        "schema": "eval_ali_run.v1",
        "runtime": {"ambient_audio_profile": {
            "asr_backend": "sherpa_sensevoice",
            "asr_model_sha256": PLACEHOLDER_ASR,
        }},
        "cases": [],
    }, indent=2) + "\n", encoding="utf-8")

    # predictions manifest: the driver reads the Sortformer hash from model.sha256.
    pred = HERE / "predictions"
    pred.mkdir(parents=True, exist_ok=True)
    (pred / "run-manifest.json").write_text(json.dumps({
        "schema": "sortformer_continuous_run.v1",
        "model": {"sha256": PLACEHOLDER_SF},
        "postprocessing_threshold": 0.30,
        "cases": [{"case_id": c} for c in CASES],
    }, indent=2) + "\n", encoding="utf-8")

    print("fixtures written to", HERE)


if __name__ == "__main__":
    main()
