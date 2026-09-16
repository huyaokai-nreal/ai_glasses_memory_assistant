"""Unit tests for ``evals/audio_frontend/build_audio_optimization_comparison.py``.

Pure, offline and fixture based: NO audio, NO ASR, NO VAD, NO MVDR, NO Sortformer,
NO network. Every fixture is generated into ``tmp_path`` at test time, so no
committed fixture can drift from the real schema.

One test (``test_real_full8_json_compatibility``) runs against the real full8
batch artefacts when they exist on disk (they are gitignored). It is skipped
when the reports directory is absent, so the suite stays green on a clean clone.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AUDIO_FRONTEND = REPO / "evals" / "audio_frontend"
if str(AUDIO_FRONTEND) not in sys.path:
    sys.path.insert(0, str(AUDIO_FRONTEND))

import summarize_continuous_full8 as summ  # noqa: E402
import build_audio_optimization_comparison as b  # noqa: E402

ASR = summ.LOCKED_ASR_MODEL_SHA256
SF = summ.LOCKED_SORTFORMER_MODEL_SHA256

CASE_FILES = b.CASE_FILES


# --------------------------------------------------------------------------- #
# Fixture builders (schema-faithful, numbers are synthetic and illustrative)
# --------------------------------------------------------------------------- #
def case_spec(case_id, *, cp_a, cp_c, ref_cp, ov_a, ov_c, ref_ov, nov_a, nov_c, ref_nov,
              der_num=100, der_den=1000, rec_num=800, rec_den=1000, finals=50, chunks=None,
              der_num_ch0=None, rec_num_ch0=None):
    return {
        "case_id": case_id,
        "cp_a": cp_a, "cp_c": cp_c, "ref_cp": ref_cp,
        "ov_a": ov_a, "ov_c": ov_c, "ref_ov": ref_ov,
        "nov_a": nov_a, "nov_c": nov_c, "ref_nov": ref_nov,
        "der_num": der_num, "der_den": der_den,
        "rec_num": rec_num, "rec_den": rec_den,
        "der_num_ch0": der_num if der_num_ch0 is None else der_num_ch0,
        "rec_num_ch0": rec_num if rec_num_ch0 is None else rec_num_ch0,
        "finals": finals,
        "chunks": finals if chunks is None else chunks,
        "processed": 600.0,
        "process": 60.0,
    }


# The locked full8 plan's exact eight meeting ids; the synthetic fixtures must use
# them verbatim so the reused check_manifest_integrity / validate_summary gates
# (which require LOCKED_CASE_IDS complete/unique/order-consistent) stay meaningful.
_RAW_SPEC_ARGS = [
    # Deliberately skewed denominators: session 0 carries most of the characters,
    # so a weighted aggregate and a mean-of-percentages differ a lot.
    dict(cp_a=100, cp_c=200, ref_cp=1000, ov_a=80, ov_c=160, ref_ov=800,
         nov_a=20, nov_c=40, ref_nov=200, der_num=180, der_den=1000, rec_num=800, rec_den=1000),
    dict(cp_a=5, cp_c=9, ref_cp=10, ov_a=4, ov_c=8, ref_ov=8,
         nov_a=1, nov_c=1, ref_nov=2, der_num=60, der_den=500, rec_num=420, rec_den=500),
    dict(cp_a=30, cp_c=60, ref_cp=300, ov_a=20, ov_c=45, ref_ov=200,
         nov_a=10, nov_c=15, ref_nov=100, der_num=90, der_den=800, rec_num=600, rec_den=800),
    dict(cp_a=40, cp_c=70, ref_cp=400, ov_a=30, ov_c=55, ref_ov=300,
         nov_a=10, nov_c=15, ref_nov=100, der_num=70, der_den=700, rec_num=500, rec_den=700),
    dict(cp_a=20, cp_c=30, ref_cp=250, ov_a=15, ov_c=25, ref_ov=180,
         nov_a=5, nov_c=5, ref_nov=70, der_num=50, der_den=600, rec_num=450, rec_den=600),
    dict(cp_a=10, cp_c=15, ref_cp=150, ov_a=8, ov_c=12, ref_ov=100,
         nov_a=2, nov_c=3, ref_nov=50, der_num=40, der_den=500, rec_num=380, rec_den=500),
    dict(cp_a=15, cp_c=25, ref_cp=200, ov_a=10, ov_c=18, ref_ov=150,
         nov_a=5, nov_c=7, ref_nov=50, der_num=45, der_den=550, rec_num=400, rec_den=550),
    dict(cp_a=8, cp_c=12, ref_cp=120, ov_a=6, ov_c=10, ref_ov=90,
         nov_a=2, nov_c=2, ref_nov=30, der_num=35, der_den=450, rec_num=330, rec_den=450),
]

DEFAULT_SPECS = [
    case_spec(case_id, **args)
    for case_id, args in zip(summ.LOCKED_CASE_IDS, _RAW_SPEC_ARGS)
]


def _intervals(errors, ref_chars):
    return {
        "intervals": 1,
        "audio_seconds": 1.0,
        "reference_chars": ref_chars,
        "errors": errors,
        "normalized_cer": (errors / ref_chars) if ref_chars else 0.0,
    }


def write_scores(path: Path, spec: dict, *, drop=(), der_ch0_override=None, rec_ch0_override=None):
    variants = {}
    for variant, cp, ov, nov in (
        ("all8", spec["cp_a"], spec["ov_a"], spec["nov_a"]),
        ("ch0", spec["cp_c"], spec["ov_c"], spec["nov_c"]),
    ):
        der_num = spec["der_num"] if variant == "all8" else spec["der_num_ch0"]
        rec_num = spec["rec_num"] if variant == "all8" else spec["rec_num_ch0"]
        if variant == "ch0" and der_ch0_override is not None:
            der_num = der_ch0_override
        if variant == "ch0" and rec_ch0_override is not None:
            rec_num = rec_ch0_override
        block = {
            "variant": variant,
            "channels": [0] if variant == "ch0" else list(range(8)),
            "segments": spec["finals"],
            "cpcer": {"errors": cp, "reference_chars": spec["ref_cp"],
                      "cer": (cp / spec["ref_cp"]) if spec["ref_cp"] else 0.0},
            "intervals": {
                "all": _intervals(cp, spec["ref_cp"]),
                "overlap": _intervals(ov, spec["ref_ov"]),
                "non_overlap": _intervals(nov, spec["ref_nov"]),
            },
            "diarization": {
                "reference_speakers": 4,
                "hypothesis_speakers": 4,
                "reference_speaker_frames": spec["der_den"],
                "errors": der_num,
                "der": der_num / spec["der_den"],
                "overlap_reference_frames": spec["rec_den"],
                "overlap_detected_frames": rec_num,
                "overlap_frame_recall": rec_num / spec["rec_den"],
            },
        }
        for key in drop:
            block.pop(key, None)
        variants[variant] = block

    checks = {
        "duplicate_identities": True,
        "at_least_one_qualified_final": True,
        "cpcer_strictly_better": True,
        "overlap_cer_strictly_better": True,
        "rtf": True,
        "network_calls_zero": True,
        "memory_unchanged": True,
        "timeline_chunk_equals_finals": True,
        "partial_not_persisted": True,
        "der_le_25": True,
        "overlap_recall_ge_70": True,
    }
    scores = {
        "schema": b.CASE_SCORE_SCHEMA,
        "created_at": "2026-09-10T00:00:00+00:00",
        "case_id": spec["case_id"],
        "processed_seconds": spec["processed"],
        "gate_scope": "full",
        "gate_scope_basis": {},
        "activity_threshold": 0.3,
        "asr_model_sha256": ASR,
        "sortformer_model_sha256": SF,
        "frontend_manifest": "",
        "frontend_rtf": 0.01,
        "full_loop_rtf": 0.1,
        "rtf_basis": "stage_sum",
        "full_process_seconds": spec["process"],
        "full_process_rtf": spec["process"] / spec["processed"],
        "input_drift": {"problems": [], "passed": True},
        "segment_integrity": {"total_rows": 1, "passed": True},
        "gold_intervals": 1,
        "gold_speakers": [],
        "variants": variants,
        "gates": {
            "baseline_variant": "ch0",
            "enhance_variant": "all8",
            "gate_scope": "full",
            "thresholds": {"rtf_max": 1.0, "der_max": 0.25, "overlap_recall_min": 0.7},
            "replay_evaluated": True,
            "enforced_checks": list(checks),
            "diagnostic_checks": [],
            "acoustic_passed": True,
            "full_e2e_passed": True,
            "checks": checks,
            "diagnostics": {},
            "passed": True,
        },
        "service_replay": {},
    }
    path.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    return scores


def write_timing(path: Path, spec: dict):
    timing = {
        "schema": b.CASE_TIMING_SCHEMA,
        "created_at": "2026-09-10T00:00:00Z",
        "out_dir": "",
        "processed_seconds": spec["processed"],
        "processed_seconds_source": "scores.json",
        "frontend_process_seconds": spec["process"] * 0.5,
        "scorer_process_seconds": spec["process"] * 0.5,
        "stage_seconds_sum": spec["process"],
        "full_process_seconds": spec["process"],
        "full_process_rtf": spec["process"] / spec["processed"],
        "rtf_max": 1.0,
        "full_process_rtf_passed": True,
    }
    path.write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")


def write_replay(path: Path, spec: dict, *, network_calls=0, memory_after=0, partial=0, ui_only=0):
    payload = {
        "isolated_home": "/tmp/isolated",
        "events_replayed": spec["finals"] + 1,
        "network_attempts": [],
        "network_calls": network_calls,
        "transcript_final_events": spec["finals"],
        "speech_rejected_events": 1,
        "partial_events": partial,
        "ui_only_events": ui_only,
        "capture_chunk_count": spec["chunks"],
        "timeline_entries_added": 0,
        "memory_before": 0,
        "memory_after": memory_after,
        "capture_status": "running",
        "persisted_audio_event_ids": [],
        "capture_status_after_finish": "interrupted",
        "audit_bytes": 1024,
        "audit_contains_embedding": False,
        "audit_contains_pcm": False,
        "chunk_texts_are_from_rejected": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def write_case(root: Path, spec: dict, **kw):
    attempt = root / spec["case_id"] / "attempt0"
    attempt.mkdir(parents=True, exist_ok=True)
    write_scores(attempt / CASE_FILES["scores"], spec, **kw)
    write_timing(attempt / CASE_FILES["timing"], spec)
    write_replay(attempt / CASE_FILES["replay"], spec)
    return attempt


def _agg(cases: list[dict], variant: str, metric: str) -> dict:
    return b.aggregate_metric(cases, variant, metric)


def make_summary(specs: list[dict], cases: list[dict]) -> dict:
    per_case_status = {
        spec["case_id"]: {
            "selected_attempt_dir": "",
            "status": {
                "evidence_valid": True,
                "quality_passed": True,
                "selected": True,
                "failure_class": "none",
                "failure_reasons": [],
            },
        }
        for spec in specs
    }
    variants = {}
    for variant in ("all8", "ch0"):
        cp = _agg(cases, variant, "cpcer")
        ov = _agg(cases, variant, "overlap")
        nov = _agg(cases, variant, "non_overlap")
        der = _agg(cases, variant, "der")
        rec = _agg(cases, variant, "recall")
        variants[variant] = {
            "cpcer": {"errors": cp["numerator"], "reference_chars": cp["denominator"],
                      "rate": cp["rate"], "aggregate_denominator_zero": False},
            "overlap": {"errors": ov["numerator"], "reference_chars": ov["denominator"],
                        "rate": ov["rate"], "aggregate_denominator_zero": False},
            "non_overlap": {"errors": nov["numerator"], "reference_chars": nov["denominator"],
                            "rate": nov["rate"], "aggregate_denominator_zero": False},
            "der": {"errors": der["numerator"], "reference_speaker_frames": der["denominator"],
                    "rate": der["rate"], "aggregate_denominator_zero": False},
            "recall": {"overlap_detected_frames": rec["numerator"],
                       "overlap_reference_frames": rec["denominator"],
                       "rate": rec["rate"], "aggregate_denominator_zero": False},
        }

    def worst(metric, key):
        ranked = b.rank_cases(cases, "all8", metric)
        return {
            "worst": [ranked[0]["case_id"], ranked[0]["rate"]],
            "ranked_worst_first": [[item["case_id"], item["rate"]] for item in ranked],
            "n_evaluable": len(ranked),
            "n_not_evaluable": 0,
            "not_evaluable_cases": [],
        }

    return {
        "schema": b.EXPECTED_SUMMARY_SCHEMA,
        "plan_id": summ.PLAN_ID,
        "n_cases": len(specs),
        "n_selected": len(specs),
        "per_case_status": per_case_status,
        "aggregate": {"variants": variants, "n_evaluable_sessions": len(specs)},
        "summary_gates": {
            "gates": {
                "aggregate_all8_cpcer_lt_ch0": True,
                "aggregate_overlap_cer_lt_ch0": True,
                "aggregate_overlap_cer_le_30": True,
                "aggregate_non_overlap_cer_le_11_5": True,
                "aggregate_der_le_25": True,
                "aggregate_overlap_recall_ge_70": True,
                "all_session_result_gates_passed": True,
                "all_evidence_privacy_integrity_passed": True,
            },
            "passed": True,
        },
        "worst_case_ranking": {
            "cpcer": worst("cpcer", "cpcer"),
            "overlap_cer": worst("overlap", "overlap_cer"),
            "non_overlap_cer": worst("non_overlap", "non_overlap_cer"),
            "der": worst("der", "der"),
            "recall": worst("recall", "recall"),
        },
        "errors": [],
        "integrity_problems": [],
    }


def make_manifest(specs: list[dict]) -> dict:
    return {
        "schema": summ.SCHEMA,
        "plan_id": summ.PLAN_ID,
        "created_at": "2026-09-10T00:00:00Z",
        "git": {
            "head_sha": "test-head-sha",
            "status": "",
            "runtime_sources_match_head": True,
            "runtime_source_sha256": {"evals/audio_frontend/x.py": "a" * 64},
        },
        "locks": {
            "prep_dir": "", "predictions": "", "source_run": "",
            "diar_variant": "micA_mean",
            "activity_threshold": 0.3,
            "enhance_variant": "all8",
            "baseline_variant": "ch0",
            "gate_scope": "full",
            "replay_timeout": 60.0,
            "model_hashes": {"asr_model_sha256": ASR, "sortformer_model_sha256": SF},
            "prep_manifest_sha256": "b" * 64,
            "prediction_run_manifest_sha256": "c" * 64,
            "source_run_manifest_sha256": "d" * 64,
            "per_case_hashes": {spec["case_id"]: {"a": "e" * 64, "b": "f" * 64} for spec in specs},
        },
        "env": {},
        "disk_budget": {},
        "cases": [{"order": i, "case_id": spec["case_id"]} for i, spec in enumerate(specs)],
        "attempts": {
            spec["case_id"]: {
                "attempts": [{"attempt": 0, "attempt_dir": f"{spec['case_id']}/attempt0",
                              "exit_code": 0, "verifier_exit": 0, "verifier_passed": True,
                              "status": {}}],
                "selected_attempt_dir": f"{spec['case_id']}/attempt0",
            }
            for spec in specs
        },
        "post_run": {
            "finished_at": "2026-09-10T00:00:00Z",
            "head_sha": "test-head-sha",
            "status": "completed",
            "code_drift": [],
            "git_status": "",
            "runtime_sources_locked": True,
            "runtime_source_sha256": {"evals/audio_frontend/x.py": "a" * 64},
        },
    }


def make_retention() -> dict:
    return {
        "schema": b.RETENTION_SCHEMA,
        "enhance_variant": "micA",
        "diar_variant": "micA_mean",
        "activity_threshold": 0.35,
        "baseline": {"overlap_cer": 0.4316, "non_overlap_cer": 0.1129},
        "oracle": {"overlap_cer": 0.2439, "non_overlap_cer": 0.1185},
        "automatic": {"overlap_cer": 0.2226, "non_overlap_cer": 0.1157, "all_cer": 0.2117,
                      "frontend_rtf": 0.0077, "cases_improved_vs_ch0": 8},
        "overlap_gain": {"oracle_absolute": 0.1878, "automatic_absolute": 0.2091, "retention": 1.11},
        "diarization_selected": {},
        "checks": {"overlap_cer": True, "cases_improved": True, "frontend_rtf": True},
        "passed": True,
        "scope": {"oracle_segment_boundaries": True, "evaluation_speaker_mapping": True,
                  "continuous_asr": False},
    }


class Pack:
    """A fully wired synthetic report pack rooted in tmp_path."""

    def __init__(self, tmp_path: Path, specs: list[dict], **case_kw):
        self.root = tmp_path
        self.specs = specs
        self.batch_dir = tmp_path / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            write_case(self.batch_dir, spec, **case_kw)
        self.out_dir = tmp_path / "out"
        self.summary_path = tmp_path / "full8-summary.json"
        self.manifest_path = tmp_path / "batch-run-manifest.json"
        self.retention_path = tmp_path / "retention.json"

    def build(self, tamper_manifest=None, tamper_summary=None, **overrides) -> dict:
        cases = [b.load_case(spec["case_id"], self.batch_dir / spec["case_id"] / "attempt0")
                 for spec in self.specs]
        summary = make_summary(self.specs, cases)
        manifest = make_manifest(self.specs)
        if tamper_summary is not None:
            tamper_summary(summary)
        if tamper_manifest is not None:
            tamper_manifest(manifest)
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.retention_path.write_text(json.dumps(make_retention()), encoding="utf-8")
        cfg = b.Config(
            full8_summary=self.summary_path,
            batch_manifest=self.manifest_path,
            batch_dir=self.batch_dir,
            auto_mask_retention=self.retention_path,
            out_dir=self.out_dir,
            **overrides,
        )
        return b.build_comparison(cfg)


@pytest.fixture()
def pack(tmp_path):
    return Pack(tmp_path, [dict(spec) for spec in DEFAULT_SPECS])


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_weighted_aggregation_not_mean_of_percentages(pack):
    """CER must be Σerrors/Σchars, never the mean of per-session percentages."""
    data = pack.build()
    row = next(r for r in data["main_table"] if r["metric"] == "cpcer")

    cases = [b.load_case(s["case_id"], pack.batch_dir / s["case_id"] / "attempt0") for s in pack.specs]
    weighted = sum(c["metrics"]["all8"]["cpcer"]["numerator"] for c in cases) / sum(
        c["metrics"]["all8"]["cpcer"]["denominator"] for c in cases
    )
    mean_of_pct = sum(c["metrics"]["all8"]["cpcer"]["rate"] for c in cases) / len(cases)

    assert row["enhanced"]["rate"] == pytest.approx(weighted)
    assert row["enhanced"]["numerator"] == sum(c["metrics"]["all8"]["cpcer"]["numerator"] for c in cases)
    assert row["enhanced"]["denominator"] == sum(c["metrics"]["all8"]["cpcer"]["denominator"] for c in cases)
    # Sanity: the two candidate computations really do differ on this fixture.
    assert abs(weighted - mean_of_pct) > 1e-3
    assert row["enhanced"]["rate"] != pytest.approx(mean_of_pct)
    assert row["enhanced"]["aggregation"].startswith("sum(numerator)/sum(denominator)")


def test_zero_denominator_rejected(tmp_path):
    specs = [dict(spec) for spec in DEFAULT_SPECS]
    specs[0]["ref_cp"] = 0
    bad = Pack(tmp_path, specs)
    with pytest.raises(b.FailClosed) as exc:
        bad.build()
    assert any("分母为零" in p for p in exc.value.problems)


def test_missing_field_rejected(tmp_path):
    p = Pack(tmp_path, [dict(spec) for spec in DEFAULT_SPECS])
    first_id = summ.LOCKED_CASE_IDS[0]
    scores_path = p.batch_dir / first_id / "attempt0" / CASE_FILES["scores"]
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores["variants"]["all8"].pop("cpcer")
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(b.FailClosed) as exc:
        p.build()
    assert any("缺少字段" in prob for prob in exc.value.problems)


def test_missing_input_file_rejected(pack):
    with pytest.raises(b.FailClosed) as exc:
        b.build_comparison(
            b.Config(
                full8_summary=pack.root / "nope.json",
                batch_manifest=pack.manifest_path,
                batch_dir=pack.batch_dir,
                auto_mask_retention=pack.retention_path,
                out_dir=pack.out_dir,
            )
        )
    assert any("输入文件不存在" in prob for prob in exc.value.problems)


def test_incomplete_full8_rejected(tmp_path):
    specs = [dict(spec) for spec in DEFAULT_SPECS][:7]
    short = Pack(tmp_path, specs)
    with pytest.raises(b.FailClosed) as exc:
        short.build()
    joined = " | ".join(exc.value.problems)
    assert "8" in joined
    assert any(("场次" in prob) or ("选满" in prob) or ("cases" in prob) for prob in exc.value.problems)


# --------------------------------------------------------------------------- #
# Targeted tests: reuse the summarizer's strict contract verbatim
# --------------------------------------------------------------------------- #
def test_manifest_v999_schema_rejected(tmp_path):
    """A schema that only differs in version tail must be rejected, not prefix-accepted."""
    p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])
    with pytest.raises(b.FailClosed) as exc:
        p.build(tamper_manifest=lambda m: m.__setitem__("schema", "continuous_full8_batch.v999"))
    assert any("schema" in prob for prob in exc.value.problems)


def test_manifest_wrong_plan_id_rejected(tmp_path):
    """Only the locked PLAN_ID is accepted; an arbitrary plan_id fails closed."""
    p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])
    with pytest.raises(b.FailClosed) as exc:
        p.build(tamper_manifest=lambda m: m.__setitem__("plan_id", "2026-01-01-wrong-plan"))
    assert any("plan_id" in prob for prob in exc.value.problems)


def test_manifest_git_status_nonempty_rejected(tmp_path):
    """A non-empty git status at report time is a hard failure (missing is NOT clean)."""
    p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])

    def tamper(m):
        m["git"]["status"] = " M evals/audio_frontend/whatever.py"

    with pytest.raises(b.FailClosed) as exc:
        p.build(tamper_manifest=tamper)
    assert any("git.status" in prob for prob in exc.value.problems)


def test_manifest_head_moved_rejected(tmp_path):
    """HEAD moving between batch start and end must fail the report."""
    p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])

    def tamper(m):
        m["post_run"]["head_sha"] = "deadbeef" * 8

    with pytest.raises(b.FailClosed) as exc:
        p.build(tamper_manifest=tamper)
    assert any("HEAD moved" in prob for prob in exc.value.problems)


def test_manifest_runtime_source_hash_drift_rejected(tmp_path):
    """Before/after runtime source hash set must be complete and identical."""
    # Case A: "after" hash set emptied -> incomplete.
    p1 = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])

    def tamper_missing(m):
        m["post_run"]["runtime_source_sha256"] = {}

    with pytest.raises(b.FailClosed) as exc:
        p1.build(tamper_manifest=tamper_missing)
    assert any("runtime source hash" in prob for prob in exc.value.problems)

    # Case B: one file's hash differs between before/after -> mismatch.
    p2 = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])

    def tamper_diff(m):
        m["post_run"]["runtime_source_sha256"]["evals/audio_frontend/x.py"] = "z" * 64

    with pytest.raises(b.FailClosed) as exc:
        p2.build(tamper_manifest=tamper_diff)
    assert any("runtime source hash" in prob for prob in exc.value.problems)


def test_seven_sessions_rejected_and_no_cli_relax(tmp_path):
    """Seven sessions are rejected, and there is no CLI knob to call it full8."""
    specs = [dict(s) for s in DEFAULT_SPECS][:7]
    short = Pack(tmp_path, specs)
    with pytest.raises(b.FailClosed) as exc:
        short.build()
    joined = " | ".join(exc.value.problems)
    assert "8" in joined
    assert any(("场次" in prob) or ("选满" in prob) or ("cases" in prob) for prob in exc.value.problems)
    # The earlier --expected-sessions knob no longer exists: 7 can never be relaxed into full8.
    assert "expected_sessions" not in [f.name for f in b.Config.__dataclass_fields__.values()]


def test_path_escape_via_selected_attempt_dir_rejected(tmp_path):
    """selected_attempt_dir must not escape the batch dir via ../, absolute, or symlink."""
    escapes = ["../../../../etc/passwd", "/etc/passwd"]
    for escape in escapes:
        p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])

        def tamper(m, sel=escape):
            m["attempts"][summ.LOCKED_CASE_IDS[0]]["selected_attempt_dir"] = sel

        with pytest.raises(b.FailClosed) as exc:
            p.build(tamper_manifest=tamper)
        assert any(("selected_attempt_dir" in prob) or ("escapes" in prob) for prob in exc.value.problems)

    # Symlink escape: a symlink inside the batch dir pointing outside it.
    p = Pack(tmp_path, [dict(s) for s in DEFAULT_SPECS])
    outside = tmp_path / "outside_target"
    outside.mkdir()
    link = p.batch_dir / "escape_link"
    link.symlink_to(outside)
    sel = "escape_link/attempt0"

    def tamper_sym(m):
        m["attempts"][summ.LOCKED_CASE_IDS[0]]["selected_attempt_dir"] = sel

    with pytest.raises(b.FailClosed) as exc:
        p.build(tamper_manifest=tamper_sym)
    assert any(("selected_attempt_dir" in prob) or ("escapes" in prob) for prob in exc.value.problems)


def test_der_is_shared_and_mislabelled_attribution_is_impossible(pack):
    data = pack.build()
    gate = data["shared_diarization_gate"]
    assert gate["shared_across_variants"] is True
    assert gate["attribution_allowed"] is False
    assert "不能写成 all8 改善了 DER" in gate["attribution_rule"]
    assert gate["der"]["rate"] == pytest.approx(gate["der"]["numerator"] / gate["der"]["denominator"])
    # The shared claim is only valid if both variants really do carry the same numbers.
    for case in (b.load_case(s["case_id"], pack.batch_dir / s["case_id"] / "attempt0") for s in pack.specs):
        assert case["metrics"]["all8"]["der"] == case["metrics"]["ch0"]["der"]


def test_der_mismatch_between_variants_rejected(tmp_path):
    p = Pack(tmp_path, [dict(spec) for spec in DEFAULT_SPECS])
    write_case(p.batch_dir, p.specs[0], der_ch0_override=p.specs[0]["der_num"] + 1)
    with pytest.raises(b.FailClosed) as exc:
        p.build()
    assert any("der" in prob and "不一致" in prob for prob in exc.value.problems)


def test_milestones_never_produce_a_delta(pack):
    data = pack.build()
    assert len(data["milestones"]) == 3
    for row in data["milestones"]:
        assert row["delta_vs_main"] is None
        assert row["comparable_to_main_table"] is False
        assert row["evidence_boundary"]
        assert row["cannot_prove"]
    # The guard itself must fire on a tampered row.
    tampered = [dict(row) for row in data["milestones"]]
    tampered[0]["delta_vs_main"] = 1.0
    with pytest.raises(b.FailClosed):
        b.assert_no_cross_milestone_delta(tampered)


def test_worst_case_ranking_is_recomputed_from_rates(tmp_path):
    specs = [dict(spec) for spec in DEFAULT_SPECS]
    # Make the 3rd locked case clearly the worst all8 cpCER without touching its denominator share.
    worst_id = summ.LOCKED_CASE_IDS[2]
    specs[2]["cp_a"] = 290
    specs[2]["ref_cp"] = 300
    p = Pack(tmp_path, specs)
    data = p.build()
    worst = data["worst_case"]
    assert worst["case_id"] == worst_id
    ranked = worst["rankings"]["cpcer"]
    assert ranked[0]["case_id"] == worst_id
    rates = [item["rate"] for item in ranked]
    assert rates == sorted(rates, reverse=True)
    assert ranked[0]["rate"] == pytest.approx(290 / 300)
    # The worst-case block reports its own metrics, not an aggregate.
    assert worst["metrics"]["cpcer"]["denominator"] == 300


def test_real_full8_json_compatibility(tmp_path):
    """Run against the real full8 artefacts when they exist (gitignored, may be absent)."""
    batch_dir = _discover_batch_dir()
    retention = _discover_retention()
    if batch_dir is None or retention is None:
        pytest.skip("真实 full8 产物不存在，跳过真实 JSON 兼容测试")

    data = b.build_comparison(
        b.Config(
            full8_summary=batch_dir / "full8-summary.json",
            batch_manifest=batch_dir / "batch-run-manifest.json",
            batch_dir=batch_dir,
            auto_mask_retention=retention,
            out_dir=tmp_path / "out",
        )
    )

    rows = {row["metric"]: row for row in data["main_table"]}
    assert rows["cpcer"]["baseline"]["numerator"] == 26655
    assert rows["cpcer"]["baseline"]["denominator"] == 81086
    assert rows["cpcer"]["enhanced"]["numerator"] == 15639
    assert rows["cpcer"]["enhanced"]["denominator"] == 81086
    assert rows["cpcer"]["enhanced"]["rate"] == pytest.approx(0.1928692992625114)
    assert rows["cpcer"]["relative_error_reduction_pct"] == pytest.approx(41.3281, abs=1e-3)
    assert rows["overlap"]["relative_error_reduction_pct"] == pytest.approx(49.7244, abs=1e-3)
    assert rows["non_overlap"]["relative_error_reduction_pct"] == pytest.approx(26.1171, abs=1e-3)
    assert data["coverage"]["n_selected"] == 8
    assert data["shared_diarization_gate"]["der"]["rate"] == pytest.approx(0.17353958127251773)
    assert data["worst_case"]["case_id"] == "R8008_M8013-full"
    # Everything we recomputed must have matched the published summary.
    assert all(item["match"] for item in data["recompute_cross_check"].values())


def test_outputs_written_and_csv_carries_main_and_per_case(pack, tmp_path):
    data = pack.build()
    written = b.write_outputs(data, pack.out_dir)
    names = {path.name for path in written}
    assert names == {"comparison.json", "metrics.csv", "REPORT.md"}
    assert "_per_case" not in json.loads((pack.out_dir / "comparison.json").read_text(encoding="utf-8"))

    with (pack.out_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scopes = {row["scope"] for row in rows}
    assert {"main", "shared_gate", "per_case"} <= scopes
    assert len([row for row in rows if row["scope"] == "per_case"]) == 8 * 2 * 3

    report = (pack.out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "多通道空间前端" in report
    assert "不是眼镜阵列或真机" in report
    assert "不等于长期归档" in report


def test_cli_main_fails_closed_without_writing(pack, capsys):
    argv = [
        "--full8-summary", str(pack.root / "missing-summary.json"),
        "--batch-manifest", str(pack.manifest_path),
        "--batch-dir", str(pack.batch_dir),
        "--auto-mask-retention", str(pack.retention_path),
        "--out-dir", str(pack.out_dir),
    ]
    assert b.main(argv) == 2
    assert not pack.out_dir.exists()


# --------------------------------------------------------------------------- #
# Real-artefact discovery helpers (no hard-coded paths in the test body)
# --------------------------------------------------------------------------- #
def _discover_batch_dir() -> Path | None:
    env = os.environ.get("AI_GLASSES_FULL8_BATCH_DIR")
    if env:
        candidate = Path(env)
        return candidate if (candidate / "full8-summary.json").is_file() else None
    for candidate in sorted((REPO / "reports" / "p4_continuous_e2e").glob("*/full8-summary.json")):
        return candidate.parent
    return None


def _discover_retention() -> Path | None:
    env = os.environ.get("AI_GLASSES_AUTO_MASK_RETENTION")
    if env:
        candidate = Path(env)
        return candidate if candidate.is_file() else None
    found = sorted((REPO / "reports" / "p3_auto_mask_score").glob("*/retention.json"))
    return found[-1] if found else None
