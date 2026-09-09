"""Aggregate eight continuous E2E sessions into one batch summary.

This module is the *single source of truth* for:
* per-attempt classification (evidence-valid vs quality-fail vs infra/evidence-abort),
* the all8/ch0 diarization consistency assertion,
* weighted (Σnumerator / Σdenominator) aggregation across the eight selected attempts,
* the batch-level summary gates, split into two tiers:
  - per-session result gates (finals / cpCER relative improvement / overlap CER
    relative improvement / internal RTF / outer RTF): a failure keeps the session
    selected and the batch continues, but the batch MUST end red via
    ``all_session_result_gates_passed``;
  - aggregate gates (DER / overlap recall / pooled CER comparisons): evaluated
    once over the pooled eight-session numerators/denominators.

It is imported by ``run_continuous_full8.py`` (the thin batch driver) so both tools
share exactly one definition of the failure state machine and the aggregation math.
It can also be run as a CLI:

    python summarize_continuous_full8.py --manifest <batch-run-manifest.json> \
        --out full8-summary.json [--strict]

Design rules (from PLAN_ID 2026-09-08-audio-continuous-e2e-full8, v3 + review 2):
* The summarizer NEVER guesses an attempt's status from its directory name or the
  wrapper exit code. It re-derives ``evidence_valid`` / ``quality_passed`` from the
  real ``scores.json`` + ``full-loop-timing.json`` fields, plus the verifier result
  the driver already recorded (a genuine WAV/segment evidence result, not a guess).
* A per-session CER scope whose denominator is 0 is ``not_evaluable`` (never labelled
  "data corrupt"); only when the *eight-session aggregate* denominator is 0 does the
  corresponding summary gate fail.
* DER = Σerrors / Σreference_speaker_frames; overlap recall =
  Σoverlap_detected_frames / Σoverlap_reference_frames. No averaging of percentages.
* DER / overlap recall never label a single session quality_fail: they exist only at
  the aggregate tier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- gate check name sets (single source of truth) -------------------------

#: Evidence-class checks: any failure => evidence failure => not selected, batch aborts.
EVIDENCE_CHECKS = (
    "network_calls_zero",          # in-application network calls
    "memory_unchanged",            # long-term memory writes during replay
    "timeline_chunk_equals_finals",  # Timeline chunk == qualified finals
    "partial_not_persisted",       # partial / UI-only events not persisted
    "duplicate_identities",        # integrity of base & candidate
)

#: Per-session *result* checks (a failure is still selected, but the batch must
#: end red via ``all_session_result_gates_passed``). DER <= 25% and overlap
#: recall >= 70% are deliberately NOT here: they are aggregate-only gates over
#: the pooled eight-session numerators/denominators and must never label a
#: single session quality_fail.
QUALITY_CHECKS = (
    "at_least_one_qualified_final",  # empty final
    "cpcer_strictly_better",        # relative CER all8 < ch0
    "overlap_cer_strictly_better",  # relative overlap CER all8 < ch0
    "rtf",                          # internal full_loop_rtf <= 1.0
)

#: Model hashes the scorer enforces. These are FIXED LOCKED constants (mirrored
#: from score_continuous_e2e.ASR_MODEL_SHA256 and diarize_sortformer.MODEL_SHA256
#: without importing those modules): a run must never self-certify the models it
#: used by reading its own manifest, so a session's recorded hashes are compared
#: against these, and the driver additionally refuses to start when the source
#: manifests disagree with the locks.
LOCKED_ASR_MODEL_SHA256 = "12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d"
LOCKED_SORTFORMER_MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"

#: SINGLE AUTHORITATIVE DEFINITION of the batch-manifest schema and plan id.
#: The driver imports these from here (it already imports this module), so the
#: writer and the validator can never drift: the driver writes exactly what this
#: module later requires. Both are matched with ``==`` (exact value, exact type);
#: a missing, mis-typed or merely differently-cased value is a hard failure.
SCHEMA = "continuous_full8_batch.v3"
PLAN_ID = "2026-09-08-audio-continuous-e2e-full8"

#: The exact eight case_ids of the locked full8 plan (mirrors the driver). The
#: summarizer enforces that a batch manifest lists EXACTLY these eight, in this
#: order, with no duplicates — running seven silently or renaming a case is a
#: batch-level integrity failure, never a relaxed green.
LOCKED_CASE_IDS = (
    "R8001_M8004-full", "R8003_M8001-full", "R8007_M8010-full", "R8007_M8011-full",
    "R8008_M8013-full", "R8009_M8018-full", "R8009_M8019-full", "R8009_M8020-full",
)

#: Tolerance for re-derived RTF comparison against the recorded value. The
#: wrapper rounds full_process_rtf to 6 dp and full_process_seconds to 3 dp; the
#: scorer rounds its RTFs to 6 dp. 1e-4 is comfortably above the rounding error
#: of those representations while still catching a real mismatch (e.g. 0.35 vs 9.0).
RTF_TOL = 1e-4

VARIANTS = ("all8", "ch0")
CER_SCOPES = ("cpcer", "overlap", "non_overlap")

# thresholds
THRESH_DER_MAX = 0.25
THRESH_RECALL_MIN = 0.70
THRESH_OVERLAP_CER_MAX = 0.30
THRESH_NON_OVERLAP_CER_MAX = 0.115
RTF_MAX = 1.0

# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def assert_diarization_equal(all8: dict, ch0: dict) -> None:
    """Assert all8 and ch0 share identical diarization frame counts.

    If they do not, the Sortformer prediction / scoring diverged between variants,
    which is an evidence failure: aggregation must stop rather than silently pick
    one side.
    """
    keys = ("errors", "reference_speaker_frames", "overlap_detected_frames", "overlap_reference_frames")
    mismatches = []
    for k in keys:
        a, b = all8.get(k), ch0.get(k)
        if a != b:
            mismatches.append(f"{k}: all8={a!r} ch0={b!r}")
    if mismatches:
        raise ValueError(
            "diarization frame counts differ between all8 and ch0 (evidence failure): "
            + "; ".join(mismatches)
        )


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _is_finite_non_negative(x: Any) -> bool:
    """True only for a real, finite, non-negative number.

    Deliberately rejects ``bool`` (a subclass of ``int`` in Python: ``True``
    is not a valid count or rate), ``NaN`` and ``Infinity`` (``math.isfinite``
    is False for both) and any non-numeric type.
    """
    if isinstance(x, bool):
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f >= 0.0


def check_numeric_consistency(scores: dict) -> List[str]:
    """Fail-closed numeric checks over scores.json.

    * every CER / DER / recall scalar is finite and non-negative;
    * ``rate == errors / reference`` where a reference exists (rate may be None
      only when the reference is 0 — the not_evaluable case handled elsewhere).
    A broken ratio (rate present but inconsistent with errors/reference) is an
    evidence failure: the aggregation math would silently double-count otherwise.
    """
    problems: List[str] = []

    def _verify_rate(where: str, errors: Any, ref: Any, rate: Any) -> None:
        if not isinstance(errors, (int, float)) or not _is_finite_non_negative(errors):
            problems.append(f"{where}: errors not a finite non-negative number")
        if not isinstance(ref, (int, float)) or not _is_finite_non_negative(ref):
            problems.append(f"{where}: reference not a finite non-negative number")
        ref_num = float(ref) if isinstance(ref, (int, float)) else 0.0
        err_num = float(errors) if isinstance(errors, (int, float)) else 0.0
        if rate is None:
            if ref_num > 0:
                problems.append(f"{where}: rate is None but reference > 0")
            return
        if not _is_finite_non_negative(rate):
            problems.append(f"{where}: rate not finite/non-negative")
            return
        if ref_num > 0 and abs(float(rate) - err_num / ref_num) > 1e-9:
            problems.append(f"{where}: rate {rate!r} != errors/reference ({err_num}/{ref_num})")

    variants = scores.get("variants") or {}
    for variant in VARIANTS:
        v = variants.get(variant) or {}
        # cpcer lives at the variant top level (errors/reference_chars/cer)
        cp = v.get("cpcer") or {}
        _verify_rate(f"variants.{variant}.cpcer", cp.get("errors"), cp.get("reference_chars"), cp.get("cer"))
        # interval CER scopes live under intervals.{overlap,non_overlap}; the
        # "all" aggregate has no independent rate to verify against.
        intervals = v.get("intervals") or {}
        for scope in ("overlap", "non_overlap"):
            blk = intervals.get(scope) or {}
            _verify_rate(f"variants.{variant}.intervals.{scope}",
                         blk.get("errors"), blk.get("reference_chars"), blk.get("normalized_cer"))
        dia = v.get("diarization") or {}
        _verify_rate(f"variants.{variant}.diarization.der",
                     dia.get("errors"), dia.get("reference_speaker_frames"), dia.get("der"))
        det, ov_ref = dia.get("overlap_detected_frames"), dia.get("overlap_reference_frames")
        rate = dia.get("overlap_frame_recall")
        if not isinstance(det, (int, float)) or not _is_finite_non_negative(det):
            problems.append(f"variants.{variant}.diarization.overlap_detected_frames invalid")
        if not isinstance(ov_ref, (int, float)) or not _is_finite_non_negative(ov_ref):
            problems.append(f"variants.{variant}.diarization.overlap_reference_frames invalid")
        ov_ref_num = float(ov_ref) if isinstance(ov_ref, (int, float)) else 0.0
        det_num = float(det) if isinstance(det, (int, float)) else 0.0
        if rate is None:
            if ov_ref_num > 0:
                problems.append(f"variants.{variant}.diarization: overlap_frame_recall is None but reference > 0")
        elif _is_finite_non_negative(rate):
            if ov_ref_num > 0 and abs(float(rate) - det_num / ov_ref_num) > 1e-9:
                problems.append(f"variants.{variant}.diarization: recall {rate!r} != detected/reference")
        else:
            problems.append(f"variants.{variant}.diarization: overlap_frame_recall not finite/non-negative")
    # In-process (internal) RTF reported by the scorer: must be present, finite
    # and non-negative. (The recorded value is re-derived against its components
    # in check_rtf_authenticity, which also flags a falsified passed flag.)
    inner = scores.get("full_loop_rtf")
    if not _is_finite_non_negative(inner):
        problems.append(f"full_loop_rtf {inner!r} missing or not finite/non-negative")
    return problems


def check_scores_schema(scores: dict, case_id: str, expected_hashes: Optional[dict]) -> List[str]:
    """Strict schema checks that are evidence failures when violated."""
    problems: List[str] = []
    got_case = scores.get("case_id")
    if got_case != case_id:
        problems.append(f"scores.case_id {got_case!r} != manifest case {case_id!r}")
    scope_top = scores.get("gate_scope")
    if scope_top != "full":
        problems.append(f"scores.gate_scope {scope_top!r} != 'full'")
    gates = scores.get("gates") or {}
    scope_gates = gates.get("gate_scope")
    if scope_gates != "full":
        problems.append(f"scores.gates.gate_scope {scope_gates!r} != 'full'")
    # Expected model hashes: BOTH must be present and must equal the fixed locks.
    # A session may never certify itself by matching whatever its own manifest
    # declares — the comparison target is the constant, not the manifest value.
    asr = scores.get("asr_model_sha256")
    sf = scores.get("sortformer_model_sha256")
    # The two model locks MUST equal the FIXED LOCKED constants. There is no
    # "pure unit test may omit the lock" escape hatch: every fixture carries the
    # real locked hashes, and a session that omits or falsifies a hash is an
    # evidence failure (a run must never self-certify its own models, and a
    # fixture that did not would be a broken test, not a relaxation).
    if asr != LOCKED_ASR_MODEL_SHA256:
        problems.append(f"scores.asr_model_sha256 {str(asr)[:12]!r} != locked {LOCKED_ASR_MODEL_SHA256[:12]}")
    if sf != LOCKED_SORTFORMER_MODEL_SHA256:
        problems.append(f"scores.sortformer_model_sha256 {str(sf)[:12]!r} != locked {LOCKED_SORTFORMER_MODEL_SHA256[:12]}")
    problems.extend(check_numeric_consistency(scores))
    return problems


def check_rtf_authenticity(scores: dict, timing: Optional[dict]) -> List[str]:
    """Re-derive outer/inner RTF from raw fields and verify the recorded passed
    flags are truthful.

    A timing that states ``full_process_rtf_passed=True`` while the real outer RTF
    exceeds the threshold (e.g. 9.0) is *rejected*: the numbers are real but the
    claim is false, which is a quality (RTF) failure, not an evidence failure. The
    summarizer must never trust a pre-computed boolean over the raw seconds.

    * outer RTF = ``full_process_seconds`` / ``processed_seconds`` (both from timing).
    * inner RTF = ``frontend_rtf`` + ``scoring_rtf`` + ``replay_rtf`` (from scores) and
      must equal the recorded ``full_loop_rtf`` (6-dp rounding tolerated).
    """
    problems: List[str] = []
    if not isinstance(timing, dict):
        return problems  # timing evidence handled by check_timing_schema

    fps = timing.get("full_process_seconds")
    ps = timing.get("processed_seconds")
    recorded = timing.get("full_process_rtf")
    passed = timing.get("full_process_rtf_passed")
    if _is_finite_non_negative(fps) and _is_finite_non_negative(ps) and float(ps) > 0:
        outer = float(fps) / float(ps)
        if _is_finite_non_negative(recorded):
            if abs(float(recorded) - outer) > RTF_TOL:
                problems.append(f"outer rtf recorded {recorded!r} != recomputed {outer:.6f}")
        elif recorded is not None:
            problems.append(f"outer rtf recorded {recorded!r} not finite/non-negative")
        if not isinstance(passed, bool):
            problems.append(f"timing.full_process_rtf_passed {passed!r} not a bool")
        elif passed != (outer <= RTF_MAX):
            problems.append(
                f"outer rtf passed flag {passed!r} inconsistent with rtf {outer:.6f} vs max {RTF_MAX}")
    else:
        problems.append("outer rtf: full_process_seconds/processed_seconds missing or invalid")

    # inner (in-process) RTF
    fr = scores.get("frontend_rtf")
    sr = scores.get("scoring_rtf")
    rr = scores.get("replay_rtf")
    inner = scores.get("full_loop_rtf")
    if _is_finite_non_negative(inner) and _is_finite_non_negative(fr) \
            and _is_finite_non_negative(sr) and _is_finite_non_negative(rr):
        recomputed = float(fr) + float(sr) + float(rr)
        if abs(recomputed - float(inner)) > RTF_TOL:
            problems.append(
                f"inner rtf recorded {inner!r} != sum {recomputed:.6f} (frontend+scoring+replay)")
        gate_rtf = _safe_get(scores, "gates", "checks", "rtf")
        if gate_rtf is True:
            if float(inner) > RTF_MAX:
                problems.append(f"inner rtf {inner!r} > max {RTF_MAX} but rtf gate passed")
        elif gate_rtf is not True:
            problems.append(f"scores.gates.checks.rtf {gate_rtf!r} (expected True for rtf<=max)")
    elif inner is not None:
        problems.append(f"inner rtf invalid: full_loop_rtf={inner!r}")
    return problems


def check_timing_schema(timing: Optional[dict], timing_path) -> List[str]:
    """Timing missing / corrupt / key-field absent / wrong type => evidence failure."""
    if timing is None:
        return ["full-loop-timing.json missing or unreadable (outer RTF evidence absent)"]
    problems: List[str] = []
    passed = timing.get("full_process_rtf_passed")
    if not isinstance(passed, bool):
        problems.append(f"timing.full_process_rtf_passed {passed!r} is not a bool")
    rtf = timing.get("full_process_rtf")
    if not isinstance(rtf, (int, float)) or not _is_finite_non_negative(rtf):
        problems.append(f"timing.full_process_rtf {rtf!r} not a finite non-negative number")
    for key in ("full_process_seconds", "processed_seconds"):
        val = timing.get(key)
        if not isinstance(val, (int, float)) or not _is_finite_non_negative(val):
            problems.append(f"timing.{key} {val!r} not a finite non-negative number")
    return problems


def verify_replay_evidence(scores: dict) -> Tuple[bool, List[str]]:
    """Independent, raw privacy / replay checks on ``scores.service_replay``.

    The scorer already folds parts of this into booleans; here we re-derive from
    the RAW replay evidence so a missing or truncated audit trail can never read
    as a pass:
    * capture terminated as ``interrupted`` (we deliberately interrupt instead of
      stopping, which would schedule the discussion archive / an LLM call);
    * audit contains no speaker embedding and no PCM payload;
    * no rejected / partial / UI-only event reached a persisted chunk (chunk text
      must never come from a rejected event);
    * final / chunk / persisted-ID counts agree and the persisted IDs are unique;
    * partial_events == 0 and ui_only_events == 0.
    Any absent or failing item is an evidence failure.
    """
    reasons: List[str] = []
    replay = scores.get("service_replay")
    if not isinstance(replay, dict):
        return False, ["evidence:service_replay absent"]
    if str(replay.get("capture_status_after_finish") or "") != "interrupted":
        reasons.append(f"replay:capture_status_after_finish={replay.get('capture_status_after_finish')!r} (expected 'interrupted')")
    if replay.get("audit_contains_embedding") is not False:
        reasons.append(f"replay:audit_contains_embedding={replay.get('audit_contains_embedding')!r}")
    if replay.get("audit_contains_pcm") is not False:
        reasons.append(f"replay:audit_contains_pcm={replay.get('audit_contains_pcm')!r}")
    if replay.get("chunk_texts_are_from_rejected") is not False:
        reasons.append(f"replay:chunk_texts_are_from_rejected={replay.get('chunk_texts_are_from_rejected')!r}")
    finals = replay.get("transcript_final_events")
    if not isinstance(finals, int) or finals < 0:
        reasons.append(f"replay:transcript_final_events={finals!r}")
    chunks = replay.get("capture_chunk_count")
    if not isinstance(chunks, int) or chunks < 0:
        reasons.append(f"replay:capture_chunk_count={chunks!r}")
    ids = list(replay.get("persisted_audio_event_ids") or [])
    valid_ids = [i for i in ids if i is not None and str(i).strip()]
    if any(i is None or not str(i).strip() for i in ids):
        reasons.append("replay:persisted ids contain None/empty entries")
    if len(valid_ids) != len(set(valid_ids)):
        reasons.append("replay:persisted ids are not unique")
    if isinstance(finals, int) and isinstance(chunks, int):
        if finals != chunks:
            reasons.append(f"replay:transcript_final_events={finals} != capture_chunk_count={chunks}")
        if finals != len(valid_ids):
            reasons.append(f"replay:transcript_final_events={finals} != n persisted ids={len(valid_ids)}")
    if replay.get("partial_events") != 0:
        reasons.append(f"replay:partial_events={replay.get('partial_events')!r} (must be 0)")
    if replay.get("ui_only_events") != 0:
        reasons.append(f"replay:ui_only_events={replay.get('ui_only_events')!r} (must be 0)")
    return (len(reasons) == 0), reasons


def classify_attempt(
    scores: dict,
    timing: Optional[dict],
    verifier_passed: bool,
    expected_hashes: Optional[dict],
    case_id: Optional[str] = None,
    timing_path=None,
) -> dict:
    """Re-derive an attempt's status from real product files.

    Returns a dict with keys:
        evidence_valid, quality_passed, selected, failure_class, failure_reasons
    ``selected`` == evidence_valid (a quality-failed but evidence-valid attempt is
    still selected into the aggregate).
    """
    reasons: List[str] = []
    evidence_ok = True
    quality_ok = True

    # --- strict schema / numeric checks are EVIDENCE checks ---
    if case_id is not None:
        schema_problems = check_scores_schema(scores, case_id, expected_hashes)
        for p in schema_problems:
            evidence_ok = False
            reasons.append(f"evidence:schema:{p}")
    tpath = timing_path
    timing_problems = check_timing_schema(timing, tpath)
    for p in timing_problems:
        evidence_ok = False
        reasons.append(f"evidence:{p}")
    replay_ok, replay_reasons = verify_replay_evidence(scores)
    if not replay_ok:
        evidence_ok = False
        reasons.extend(replay_reasons)

    gates = scores.get("gates") or {}
    checks = gates.get("checks") or {}

    # --- evidence class (from gates.checks + aux) ---
    for name in EVIDENCE_CHECKS:
        val = checks.get(name)
        if val is not True:
            evidence_ok = False
            reasons.append(f"evidence:{name}={val!r}")

    if gates.get("replay_evaluated") is not True:
        evidence_ok = False
        reasons.append("evidence:replay_not_evaluated")

    seg = _safe_get(scores, "segment_integrity", "passed")
    if seg is not True:
        evidence_ok = False
        reasons.append(f"evidence:segment_integrity.passed={seg!r}")

    drift = _safe_get(scores, "input_drift", "passed")
    if drift is not True:
        evidence_ok = False
        reasons.append(f"evidence:input_drift.passed={drift!r}")

    # WAV/segment evidence must be a real, recorded verifier pass. Missing or
    # failed => fail-closed (never assumed passed).
    if not verifier_passed:
        evidence_ok = False
        reasons.append("evidence:verifier_failed")

    # --- quality class (from gates.checks + outer RTF) ---
    for name in QUALITY_CHECKS:
        val = checks.get(name)
        if val is not True:
            quality_ok = False
            reasons.append(f"quality:{name}={val!r}")

    # --- RTF authenticity: re-derive from raw seconds, do NOT trust the flags ---
    rtf_problems = check_rtf_authenticity(scores, timing)
    for p in rtf_problems:
        quality_ok = False
        reasons.append(f"quality:rtf:{p}")
    # A legitimately exceeded outer RTF (flag False, consistent) is still a quality fail
    outer_rtf_passed = bool(_safe_get(timing, "full_process_rtf_passed")) if timing else False
    if not outer_rtf_passed:
        quality_ok = False
        reasons.append("quality:full_process_rtf_passed=False")

    if evidence_ok:
        failure_class = "none" if quality_ok else "quality_fail"
    else:
        # Without evidence we cannot trust the product; prefer evidence_fail, but a
        # product-missing situation (caller passes scores=None) is recorded as infra_fail.
        failure_class = "evidence_fail"

    return {
        "evidence_valid": evidence_ok,
        "quality_passed": quality_ok,
        "selected": evidence_ok,
        "failure_class": failure_class,
        "failure_reasons": reasons,
    }


def classify_missing_product(reason: str) -> dict:
    """Status for an attempt whose scores.json / timing is absent or unparseable.

    A session that ran but whose product is missing/corrupt is an infra failure —
    the driver treats it as non-retryable. A case with NO selected attempt at all
    (the session never produced one) is an evidence failure and is handled by the
    caller before products are even read.
    """
    return {
        "evidence_valid": False,
        "quality_passed": False,
        "selected": False,
        "failure_class": "infra_fail",
        "failure_reasons": [f"infra:product_missing:{reason}"],
    }


def _cer_pair(scores: dict, variant: str, scope: str) -> Tuple[int, int, Optional[float], bool]:
    """Return (errors, reference_chars, cer, evaluable) for a CER scope.

    ``evaluable`` is False when reference_chars == 0 (=> not_evaluable, never a
    divide-by-zero / "data corrupt" label).
    """
    if scope == "cpcer":
        block = _safe_get(scores, "variants", variant, "cpcer") or {}
    else:
        block = _safe_get(scores, "variants", variant, "intervals", scope) or {}
    errors = block.get("errors") or 0
    ref = block.get("reference_chars") or 0
    cer = block.get("cer") if scope == "cpcer" else block.get("normalized_cer")
    evaluable = ref > 0
    return int(errors), int(ref), (float(cer) if cer is not None else None), evaluable


def _session_metrics(scores: dict) -> dict:
    """Per-session metrics with not_evaluable flags per scope.

    Every metric block — CER scopes, DER and overlap recall, per-session as well as
    aggregate — exposes its scalar under the SAME key ``"rate"``. Keeping one key name
    is deliberate: ``worst_case_ranking`` and the gate evaluation read blocks
    generically, and an earlier per-session/aggregate key mismatch silently emptied the
    DER / recall worst-case ranking.
    """
    out: Dict[str, Any] = {"variants": {}}
    for variant in VARIANTS:
        v: Dict[str, Any] = {}
        for scope in CER_SCOPES:
            errors, ref, cer, evaluable = _cer_pair(scores, variant, scope)
            v[scope] = {
                "errors": errors,
                "reference_chars": ref,
                "rate": cer,
                "evaluable": evaluable,
                "not_evaluable": not evaluable,
            }
        dia = _safe_get(scores, "variants", variant, "diarization") or {}
        dia_errors = int(dia.get("errors") or 0)
        ref_spk = int(dia.get("reference_speaker_frames") or 0)
        det = int(dia.get("overlap_detected_frames") or 0)
        ov_ref = int(dia.get("overlap_reference_frames") or 0)
        v["der"] = {
            "errors": dia_errors,
            "reference_speaker_frames": ref_spk,
            "rate": (dia_errors / ref_spk) if ref_spk > 0 else None,
            "evaluable": ref_spk > 0,
            "not_evaluable": ref_spk <= 0,
        }
        v["recall"] = {
            "overlap_detected_frames": det,
            "overlap_reference_frames": ov_ref,
            "rate": (det / ov_ref) if ov_ref > 0 else None,
            "evaluable": ov_ref > 0,
            "not_evaluable": ov_ref <= 0,
        }
        out["variants"][variant] = v
    return out


def aggregate_sessions(sessions: List[dict]) -> dict:
    """Weighted aggregation over the selected (evidence-valid) sessions.

    Each ``session`` is ``{"case_id": str, "metrics": <_session_metrics output>}``.
    Skips not_evaluable scopes per-session. Returns aggregate numerators/denominators
    and rates (None when the aggregate denominator is 0).
    """
    agg: Dict[str, Any] = {"variants": {}, "n_evaluable_sessions": len(sessions)}
    for variant in VARIANTS:
        v: Dict[str, Any] = {}
        for scope in CER_SCOPES:
            num = den = 0
            for s in sessions:
                blk = s["metrics"]["variants"][variant][scope]
                if blk["evaluable"]:
                    num += blk["errors"]
                    den += blk["reference_chars"]
            v[scope] = {
                "errors": num,
                "reference_chars": den,
                "rate": (num / den) if den > 0 else None,
                "aggregate_denominator_zero": den == 0,
            }
        # DER / recall come from all8 diarization (asserted equal to ch0).
        d_num = d_den = 0
        r_det = r_ref = 0
        for s in sessions:
            d = s["metrics"]["variants"][variant]["der"]
            if d["evaluable"]:
                d_num += d["errors"]
                d_den += d["reference_speaker_frames"]
            r = s["metrics"]["variants"][variant]["recall"]
            if r["evaluable"]:
                r_det += r["overlap_detected_frames"]
                r_ref += r["overlap_reference_frames"]
        v["der"] = {
            "errors": d_num,
            "reference_speaker_frames": d_den,
            "rate": (d_num / d_den) if d_den > 0 else None,
            "aggregate_denominator_zero": d_den == 0,
        }
        v["recall"] = {
            "overlap_detected_frames": r_det,
            "overlap_reference_frames": r_ref,
            "rate": (r_det / r_ref) if r_ref > 0 else None,
            "aggregate_denominator_zero": r_ref == 0,
        }
        agg["variants"][variant] = v
    return agg


def evaluate_summary_gates(
    agg: dict,
    sessions: List[dict],
    all_evidence_valid: bool,
) -> dict:
    """Evaluate the batch summary gates over the pooled eight-session numbers.

    All aggregate rates use Σnum/Σden. The per-session result gates are evaluated
    per session in ``collect_selected`` and folded into ``sessions`` as
    ``result_gates_passed``; the batch-level gate ``all_session_result_gates_passed``
    requires EVERY selected session to have passed them (a session with a failed
    finals / cpCER / overlap-CER / internal-RTF / outer-RTF gate makes the batch
    red even when the pooled aggregate still improves).
    """
    a8 = agg["variants"]["all8"]
    c0 = agg["variants"]["ch0"]

    # Any rate that is None means its aggregate denominator was 0 -> fail-closed
    # (None is treated as a FAIL, never as a pass).
    a8c, c0c = a8["cpcer"]["rate"], c0["cpcer"]["rate"]
    g1 = bool(a8c is not None and c0c is not None and a8c < c0c)
    a8o, c0o = a8["overlap"]["rate"], c0["overlap"]["rate"]
    g2 = bool(a8o is not None and c0o is not None and a8o < c0o)
    g3 = bool(a8["overlap"]["rate"] is not None and a8["overlap"]["rate"] <= THRESH_OVERLAP_CER_MAX)
    g4 = bool(a8["non_overlap"]["rate"] is not None and a8["non_overlap"]["rate"] <= THRESH_NON_OVERLAP_CER_MAX)
    g5 = bool(a8["der"]["rate"] is not None and a8["der"]["rate"] <= THRESH_DER_MAX)
    g6 = bool(a8["recall"]["rate"] is not None and a8["recall"]["rate"] >= THRESH_RECALL_MIN)
    # gate 7: EVERY selected session passed its per-session result gates
    # (finals / cpCER relative / overlap relative / internal RTF / outer RTF).
    g7 = all(s.get("result_gates_passed") is True for s in sessions)
    # gate 8: all evidence/privacy/integrity passed
    g8 = all_evidence_valid

    gates = {
        "aggregate_all8_cpcer_lt_ch0": g1,
        "aggregate_overlap_cer_lt_ch0": g2,
        "aggregate_overlap_cer_le_30": g3,
        "aggregate_non_overlap_cer_le_11_5": g4,
        "aggregate_der_le_25": g5,
        "aggregate_overlap_recall_ge_70": g6,
        "all_session_result_gates_passed": g7,
        "all_evidence_privacy_integrity_passed": g8,
    }
    passed = all(v is True for v in gates.values())
    return {"gates": gates, "passed": passed}


#: (ranking name, key inside a per-session variant block, worse_is_higher)
RANKING_METRICS = (
    ("cpcer", "cpcer", True),
    ("overlap_cer", "overlap", True),
    ("non_overlap_cer", "non_overlap", True),
    ("der", "der", True),
    ("recall", "recall", False),  # recall: LOWER is worse
)


def worst_case_ranking(sessions: List[dict]) -> dict:
    """Rank sessions per metric, skipping not_evaluable, single out the worst.

    ``ranked`` is ordered worst-first so the three known sub-70% recall sessions are
    listed explicitly even when the eight-session aggregate recall passes. Recall is
    ranked ascending (lower = worse); every other metric descending (higher = worse).
    """
    ranking: Dict[str, Any] = {}
    for metric_name, block_key, worse_is_higher in RANKING_METRICS:
        rows = []
        not_evaluable = []
        for s in sessions:
            blk = s["metrics"]["variants"]["all8"][block_key]
            if blk.get("not_evaluable") or blk.get("rate") is None:
                not_evaluable.append(s["case_id"])
                continue
            rows.append((s["case_id"], blk["rate"]))
        rows.sort(key=lambda x: x[1], reverse=worse_is_higher)
        ranking[metric_name] = {
            "worst": rows[0] if rows else None,
            "ranked_worst_first": rows,
            "n_evaluable": len(rows),
            "n_not_evaluable": len(not_evaluable),
            "not_evaluable_cases": not_evaluable,
        }
    return ranking


def build_full_summary(
    manifest: dict,
    sessions: List[dict],
    per_case_status: Dict[str, dict],
    agg: dict,
    gates_out: dict,
    ranking: dict,
) -> dict:
    return {
        "schema": "continuous_full8_summary.v1",
        "plan_id": manifest.get("plan_id"),
        "n_cases": len(manifest.get("cases", [])),
        "n_selected": len(sessions),
        "per_case_status": per_case_status,
        "aggregate": agg,
        "summary_gates": gates_out,
        "worst_case_ranking": ranking,
    }


def resolve_attempt_dir(batch_root: Path, sel: Any) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve ``sel`` against ``batch_root`` and refuse anything escaping it.

    The driver stores relative ``<case_id>/attemptN`` paths in the manifest. A
    later consumer (e.g. this summarizer reading a copied manifest elsewhere)
    resolves against ITS batch root and must never read outside it — so after
    ``resolve()`` (which also defeats symlinks) the result must still live under
    the batch root. Absolute legacy paths are still accepted only while they
    resolve inside the batch root.
    """
    if not sel:
        return None, "no selected_attempt_dir"
    batch_abs = batch_root.resolve()
    raw = Path(str(sel))
    candidate = (batch_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        inside = candidate == batch_abs or batch_abs in candidate.parents
    except ValueError:
        inside = False
    if not inside:
        return None, f"selected_attempt_dir escapes batch root: {sel}"
    return candidate, None


def check_manifest_integrity(manifest: dict) -> List[str]:
    """Strict batch-level checks the summarizer entry enforces before trusting a
    summary. Any problem makes the batch red (non-zero exit). Covers:

    * ``schema`` EXACTLY ``continuous_full8_batch.v3`` and ``plan_id`` EXACTLY the
      locked plan id (missing / wrong type / wrong value all fail);
    * ``git.status`` EXPLICITLY the empty string (missing is NOT clean) and
      ``git.runtime_sources_match_head`` strictly ``True``;
    * exactly eight ordered, unique cases (the locked plan, no 7-case runs);
    * every case has a selected attempt (an evidence gap otherwise);
    * ``post_run`` completed, ``finished_at`` present, and ``code_drift`` EXPLICITLY
      empty (a missing code_drift is NOT treated as clean);
    * HEAD before == HEAD after; runtime sources still locked at batch end;
    * the before/after runtime source hash sets are complete, identical, and
      per-file equal (missing file / missing hash / mismatch all fail);
    * ``git_status`` EXPLICITLY the empty string (missing is NOT treated as clean);
    * the two model locks complete and equal to the fixed constants.
    """
    problems: List[str] = []

    # --- schema / plan_id: exact value AND type, never a substring or "close enough"
    schema = manifest.get("schema")
    if not isinstance(schema, str) or schema != SCHEMA:
        problems.append(f"manifest.schema {schema!r} != {SCHEMA!r}")
    plan_id = manifest.get("plan_id")
    if not isinstance(plan_id, str) or plan_id != PLAN_ID:
        problems.append(f"manifest.plan_id {plan_id!r} != {PLAN_ID!r}")

    # --- git block: status explicitly empty, sources explicitly HEAD-clean -------
    git_block = manifest.get("git") or {}
    if not isinstance(git_block, dict):
        problems.append(f"manifest.git missing or not an object: {git_block!r}")
    else:
        git_status_before = git_block.get("status")
        if not isinstance(git_status_before, str):
            problems.append(
                f"git.status missing or not a string: {git_status_before!r}")
        elif git_status_before != "":
            problems.append(f"git.status non-empty: {git_status_before!r}")
        match_head = git_block.get("runtime_sources_match_head")
        if match_head is not True:
            problems.append(
                f"git.runtime_sources_match_head {match_head!r} is not strictly True")

    cases = manifest.get("cases") or []
    ids = [c.get("case_id") for c in cases]
    if len(ids) != 8:
        problems.append(f"manifest has {len(ids)} cases, expected exactly 8")
    if len(ids) != len(set(ids)):
        problems.append("duplicate case_ids in manifest")
    if list(ids) != list(LOCKED_CASE_IDS):
        problems.append("case_ids not exactly the locked eight in order")
    attempts = manifest.get("attempts") or {}
    for cid in ids:
        rec = attempts.get(cid) or {}
        if not rec.get("selected_attempt_dir"):
            problems.append(f"{cid}: no selected_attempt_dir (evidence gap)")

    post = manifest.get("post_run") or {}
    if not isinstance(post, dict):
        return ["post_run missing or not an object"]
    if post.get("status") != "completed":
        problems.append(f"post_run.status {post.get('status')!r} != completed")
    finished_at = post.get("finished_at")
    if not isinstance(finished_at, str) or not finished_at.strip():
        problems.append(f"post_run.finished_at missing or not a non-empty string: {finished_at!r}")
    # code_drift must be EXPLICITLY an EMPTY LIST. A missing field (None) is NOT
    # "clean" — it means the driver never recorded a drift verdict, which is itself a
    # failure. A non-list (e.g. "" or False) is a type violation, also a failure.
    drift = post.get("code_drift")
    if not isinstance(drift, list):
        problems.append(f"post_run.code_drift missing or not a list: {drift!r}")
    elif drift:
        problems.append(f"post_run.code_drift non-empty: {drift!r}")

    head_before = (manifest.get("git") or {}).get("head_sha")
    head_after = post.get("head_sha")
    if not head_before or not head_after:
        problems.append("missing HEAD sha (before/after)")
    elif head_before != head_after:
        problems.append(f"HEAD moved during batch: {str(head_before)[:12]} -> {str(head_after)[:12]}")

    if post.get("runtime_sources_locked") is not True:
        problems.append("runtime sources not locked/clean at batch end")

    # Before/after runtime source hash sets must be COMPLETE, the SAME set, and have
    # identical per-file values. A missing file, a missing hash, or any mismatch is a
    # non-zero failure (this catches mid-batch code drift that slipped past the lock).
    before = (manifest.get("git") or {}).get("runtime_source_sha256") or {}
    after = post.get("runtime_source_sha256") or {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        problems.append("runtime source hash set malformed (before/after)")
    elif not before or not after:
        problems.append("runtime source hash set incomplete (before or after empty)")
    else:
        only_before = sorted(set(before) - set(after))
        only_after = sorted(set(after) - set(before))
        if only_before or only_after:
            problems.append(
                f"runtime source hash set differs: only_before={only_before} only_after={only_after}")
        else:
            for rel in before:
                bv, av = before.get(rel), after.get(rel)
                if not bv or not av or bv != av:
                    problems.append(f"runtime source hash mismatch: {rel}")

    locks = manifest.get("locks") or {}
    mh = locks.get("model_hashes") or {}
    if not mh.get("asr_model_sha256") or not mh.get("sortformer_model_sha256"):
        problems.append("manifest model locks missing")
    elif (mh.get("asr_model_sha256") != LOCKED_ASR_MODEL_SHA256
          or mh.get("sortformer_model_sha256") != LOCKED_SORTFORMER_MODEL_SHA256):
        problems.append("manifest model locks != fixed constants")

    # git_status must be EXPLICITLY the empty string (clean). A missing field (None)
    # must NOT be treated as clean — absence of a recorded status is a failure.
    git_status = post.get("git_status")
    if not isinstance(git_status, str):
        problems.append(f"post_run.git_status missing or not a string: {git_status!r}")
    elif git_status != "":
        problems.append(f"post_run.git_status non-empty: {git_status!r}")
    return problems


def collect_selected(manifest: dict, batch_root: Path, expected_hashes: Optional[dict]) -> Tuple[List[dict], Dict[str, dict], List[str]]:
    """Read each case's selected_attempt_dir, load products, classify, build session list.

    Returns (sessions, per_case_status, errors). ``errors`` is non-empty when a case
    has no selected attempt (evidence gap) or a product file is unreadable.
    """
    sessions: List[dict] = []
    per_case_status: Dict[str, dict] = {}
    errors: List[str] = []
    batch_root = Path(batch_root)

    for case in manifest.get("cases", []):
        cid = case["case_id"]
        rec = manifest.get("attempts", {}).get(cid, {})
        sel = rec.get("selected_attempt_dir")
        attempt_dir, resolve_err = resolve_attempt_dir(batch_root, sel)
        if resolve_err:
            # No selected attempt / escaping path: the session never produced a
            # usable, in-batch product -> evidence gap -> evidence_fail (aborts).
            errors.append(f"{cid}: {resolve_err}")
            per_case_status[cid] = {"selected_attempt_dir": sel, "status": {
                "evidence_valid": False, "quality_passed": False, "selected": False,
                "failure_class": "evidence_fail",
                "failure_reasons": [f"evidence:{resolve_err}"],
            }}
            continue
        scores_path = attempt_dir / "scores.json"
        timing_path = attempt_dir / "full-loop-timing.json"
        if not scores_path.is_file():
            st = classify_missing_product("scores.json missing")
            errors.append(f"{cid}: scores.json missing at {scores_path}")
            per_case_status[cid] = {"selected_attempt_dir": str(attempt_dir), "status": st}
            continue
        try:
            scores = read_json(scores_path)
        except (ValueError, OSError) as exc:
            st = classify_missing_product(f"scores.json unparseable: {exc}")
            errors.append(f"{cid}: scores.json unparseable")
            per_case_status[cid] = {"selected_attempt_dir": str(attempt_dir), "status": st}
            continue
        timing = None
        if timing_path.is_file():
            try:
                timing = read_json(timing_path)
            except (ValueError, OSError):
                timing = None
        # Verifier result recorded by the driver in the attempt record. Fail-closed:
        # if no attempt record matches the selected dir, or it has no verifier result,
        # the WAV/segment evidence is UNVERIFIED and must not be assumed to pass.
        # The selected dir is matched by the manifest's own relative path (the
        # ``attempt_dir`` field) — exact string equality, never a guessed join.
        attempt_records = rec.get("attempts", [])
        verifier_passed = False
        for a in attempt_records:
            if a.get("attempt_dir") == sel:
                verifier_passed = a.get("verifier_passed") is True
                break
        status = classify_attempt(scores, timing, verifier_passed, expected_hashes,
                                  case_id=cid, timing_path=timing_path)
        per_case_status[cid] = {"selected_attempt_dir": str(attempt_dir), "status": status}

        # fail-closed: an attempt recorded by the driver as "selected" MUST re-derive
        # as evidence-valid here. If it does not, it is not aggregated (never trust
        # the directory name or the wrapper exit code).
        if not status["selected"]:
            errors.append(f"{cid}: selected attempt is not evidence-valid ({status['failure_class']})")
            continue

        # diarization consistency assert
        try:
            all8_d = _safe_get(scores, "variants", "all8", "diarization") or {}
            ch0_d = _safe_get(scores, "variants", "ch0", "diarization") or {}
            assert_diarization_equal(all8_d, ch0_d)
        except ValueError as exc:
            errors.append(f"{cid}: {exc}")
            continue

        # Per-session result gates: finals / cpCER relative / overlap relative /
        # internal RTF (all from gates.checks) + outer RTF (from timing). DER /
        # overlap recall are NOT evaluated here — aggregate tier only.
        result_ok = status["quality_passed"] and bool(_safe_get(scores, "gates", "checks", "rtf")) \
            and (bool(_safe_get(timing, "full_process_rtf_passed")) if timing else False)
        sessions.append({
            "case_id": cid,
            "metrics": _session_metrics(scores),
            "result_gates_passed": result_ok,
            "internal_rtf_passed": bool(_safe_get(scores, "gates", "checks", "rtf")),
            "outer_rtf_passed": bool(_safe_get(timing, "full_process_rtf_passed")) if timing else False,
        })

    return sessions, per_case_status, errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="non-zero exit when any summary gate fails")
    args = parser.parse_args(argv)

    manifest = read_json(args.manifest)
    batch_root = args.manifest.parent
    expected_hashes = _safe_get(manifest, "locks", "model_hashes")

    sessions, per_case_status, errors = collect_selected(manifest, batch_root, expected_hashes)
    all_evidence_valid = all(
        s["status"]["evidence_valid"] for s in per_case_status.values()
    ) and not errors

    # --- strict batch-level integrity (group 1): one / seven / duplicate / aborted
    # / drifted / missing model lock all force a red, non-zero-exit summary ---
    integrity_problems = check_manifest_integrity(manifest)
    if integrity_problems:
        errors.extend(integrity_problems)
        all_evidence_valid = False

    agg = aggregate_sessions(sessions)
    gates_out = evaluate_summary_gates(agg, sessions, all_evidence_valid)
    ranking = worst_case_ranking(sessions)
    summary = build_full_summary(manifest, sessions, per_case_status, agg, gates_out, ranking)
    summary["errors"] = errors
    summary["integrity_problems"] = integrity_problems

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_gates": gates_out["gates"],
        "passed": gates_out["passed"],
        "n_selected": len(sessions),
        "errors": errors,
    }, ensure_ascii=False, indent=2))

    # A batch is only green when EVERY gate (including batch integrity) passes.
    if not gates_out["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
