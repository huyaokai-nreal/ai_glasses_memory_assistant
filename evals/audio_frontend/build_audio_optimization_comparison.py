#!/usr/bin/env python3
"""Build a reviewable AliMeeting audio-frontend optimization report pack.

Scope (read-only):
    This script NEVER runs ASR / VAD / MVDR / Sortformer / continuous E2E audio and
    never calls a network or cloud model. It reads pre-existing JSON artefacts,
    re-aggregates every rate from raw numerators/denominators, and writes
    ``comparison.json`` / ``metrics.csv`` / ``REPORT.md`` into an explicitly
    passed output directory.

Comparability rules enforced here:
    1. The main table compares only ``ch0`` vs ``all8`` inside ONE full8 batch:
       same 8 meetings, same diarization tracks, same ASR, same gold, same gates.
    2. Every rate is recomputed as sum(numerator) / sum(denominator). Averaging
       per-session percentages is not implemented and cannot happen by accident.
    3. DER and overlap recall come from the shared diarization track, so they are
       labelled as a shared gate and are never attributed to the all8 variant.
    4. oracle / auto-mask / full8 E2E are three milestone rows with different
       evidence boundaries; no delta, ranking or trend is derived across them.

Fail-closed:
    Missing input, unexpected schema, missing field, zero denominator, an
    incomplete full8 selection, a failed gate or a failed re-computation
    cross-check aborts with a non-zero exit and writes NOTHING (no "passed"
    report can ever be produced from a broken input set).

Usage example (all paths are CLI arguments; nothing is hard-coded here):

    python evals/audio_frontend/build_audio_optimization_comparison.py \
        --full8-summary  <batch>/full8-summary.json \
        --batch-manifest <batch>/batch-run-manifest.json \
        --batch-dir      <batch> \
        --auto-mask-retention <dir>/retention.json \
        --sortformer-scores   <dir>/scores.json \
        --out-dir       <gitignored dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Gate names and model locks are imported from the summarizer so this report can
# never drift from the pipeline's own definition of a failure. In particular
# DER <= 25% and overlap recall >= 70% are AGGREGATE-ONLY gates there, so a single
# session below them is a disclosed caveat here, never a per-session failure.
# Gate names and model locks are imported from the summarizer so this report can
# never drift from the pipeline's own definition of a failure. In particular
# DER <= 25% and overlap recall >= 70% are AGGREGATE-ONLY gates there, so a single
# session below them is a disclosed caveat here, never a per-session failure.
# We also reuse the summarizer's STRICT, SINGLE-SOURCE-OF-TRUTH batch contract
# (SCHEMA / PLAN_ID / LOCKED_CASE_IDS / check_manifest_integrity /
# resolve_attempt_dir): this report must not invent a weaker parallel validator.
from summarize_continuous_full8 import (  # type: ignore  # noqa: E402
    EVIDENCE_CHECKS,
    LOCKED_ASR_MODEL_SHA256,
    LOCKED_SORTFORMER_MODEL_SHA256,
    QUALITY_CHECKS,
    SCHEMA,
    PLAN_ID,
    LOCKED_CASE_IDS,
    check_manifest_integrity,
    resolve_attempt_dir,
)

SCHEMA_OUT = "audio_optimization_comparison.v1"

#: Aggregate-only gate names: evaluated once over the pooled eight sessions.
AGGREGATE_ONLY_CHECKS = ("der_le_25", "overlap_recall_ge_70")

# --- Input schema locks. A different producer means we cannot vouch for fields. ---
#: The full8 batch contract is OWNED by the summarizer. We import its locked
#: SCHEMA / PLAN_ID / LOCKED_CASE_IDS and call its strict check_manifest_integrity
#: verbatim — never a prefix match, never an arbitrary case set, never a weaker
#: re-implementation.
EXPECTED_SUMMARY_SCHEMA = "continuous_full8_summary.v1"
RETENTION_SCHEMA = "eval_ali_auto_mask_retention.v1"
CASE_SCORE_SCHEMA = "eval_ali_continuous_e2e_score.v1"
CASE_TIMING_SCHEMA = "continuous_full_loop_timing.v1"
SORTFORMER_SCORE_SCHEMA = "eval_ali_diarization_score.v1"
ORACLE_NOTE_SCHEMA = "audio_frontend_oracle_milestone.v1"

#: full8 is a locked, eight-meeting plan. The report refuses any other count, so a
#: 7-meeting run can never be silently green. REQUIRED_SESSIONS is derived from the
#: same LOCKED_CASE_IDS the summarizer enforces, keeping both tools in lockstep.
REQUIRED_SESSIONS = len(LOCKED_CASE_IDS)
DEFAULT_ORACLE_NOTE = "PLANS 历史记录，无机器可读源，不参与计算"

MAIN_METRICS = ("cpcer", "overlap", "non_overlap")
SHARED_GATE_METRICS = ("der", "recall")
VARIANTS = ("all8", "ch0")
BASELINE_VARIANT = "ch0"
ENHANCED_VARIANT = "all8"

# metric -> (path inside the per-variant block, numerator key, denominator key)
METRIC_SOURCES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "cpcer": (("cpcer",), "errors", "reference_chars"),
    "overlap": (("intervals", "overlap"), "errors", "reference_chars"),
    "non_overlap": (("intervals", "non_overlap"), "errors", "reference_chars"),
    "der": (("diarization",), "errors", "reference_speaker_frames"),
    "recall": (("diarization",), "overlap_detected_frames", "overlap_reference_frames"),
}

METRIC_LABELS = {
    "cpcer": "cpCER（拼接最小编辑距离字错率）",
    "overlap": "overlap CER（重叠语音区间字错率）",
    "non_overlap": "non-overlap CER（非重叠语音区间字错率）",
    "der": "DER（分音错误率）",
    "recall": "overlap recall（重叠帧召回）",
}

CASE_FILES = {
    "scores": "scores.json",
    "timing": "full-loop-timing.json",
    "replay": "service-replay.json",
}


class FailClosed(Exception):
    """Raised when the input set cannot support a trustworthy report.

    The runner turns this into a non-zero exit with no artefacts written.
    """

    def __init__(self, problems: list[str]):
        self.problems = [str(p) for p in problems]
        super().__init__("; ".join(self.problems))


class ZeroDenominator(FailClosed):
    """A rate was requested with a missing or zero denominator."""


@dataclass(frozen=True)
class Config:
    full8_summary: Path
    batch_manifest: Path
    batch_dir: Path
    auto_mask_retention: Path
    out_dir: Path
    sortformer_scores: Optional[Path] = None
    oracle_json: Optional[Path] = None
    oracle_note: str = DEFAULT_ORACLE_NOTE


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FailClosed([f"{label}: 输入文件不存在 -> {path}"])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FailClosed([f"{label}: JSON 解析失败 -> {path}: {exc}"]) from exc
    if not isinstance(data, dict):
        raise FailClosed([f"{label}: 顶层不是 JSON 对象 -> {path}"])
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dig(obj: Any, path: tuple[str, ...], where: str) -> Any:
    """Fetch a nested field or fail-closed. Never returns a silent default."""
    current = obj
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(current, dict) or key not in current:
            raise FailClosed([f"缺少字段: {where}.{'.'.join(walked)}"])
        current = current[key]
    return current


def as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FailClosed([f"字段不是数值: {where} -> {value!r}"])
    return float(value)


def ratio(numerator: float, denominator: float, where: str) -> float:
    """Every rate in this report goes through here: a zero denominator is fatal."""
    if denominator is None or denominator == 0:
        raise ZeroDenominator([f"{where}: 分母为零或缺失 (numerator={numerator}, denominator={denominator})"])
    return numerator / denominator


def pct(rate: float) -> float:
    return rate * 100.0


def fmt_pct(rate: float) -> str:
    return f"{pct(rate):.2f}%"


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


def nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def gitignore_state(out_dir: Path) -> dict:
    """The report pack must live in a gitignored directory; verify when possible."""
    anchor = nearest_existing(out_dir)
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(anchor),
            capture_output=True,
            text=True,
        )
    except OSError:
        return {"checked": False, "ignored": None, "note": "git 不可用，跳过 gitignore 校验"}
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "checked": False,
            "ignored": None,
            "note": "输出目录不在 git 工作树内，跳过 gitignore 校验",
        }
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(out_dir)],
        cwd=str(anchor),
        capture_output=True,
        text=True,
    )
    if ignored.returncode == 0:
        return {"checked": True, "ignored": True, "note": ""}
    if ignored.returncode == 1:
        raise FailClosed([f"输出目录未被 gitignore，拒绝产出报告: {out_dir}"])
    return {
        "checked": False,
        "ignored": None,
        "note": f"git check-ignore 异常 (rc={ignored.returncode})，跳过校验",
    }


# --------------------------------------------------------------------------- #
# Input loading / validation
# --------------------------------------------------------------------------- #
def validate_summary(summary: dict) -> dict:
    schema = summary.get("schema")
    if schema != EXPECTED_SUMMARY_SCHEMA:
        raise FailClosed([f"full8-summary schema 不匹配: 期望 {EXPECTED_SUMMARY_SCHEMA}, 实际 {schema!r}"])

    # The summary must carry the locked plan id (exact match, no prefix / near-match).
    plan_id = summary.get("plan_id")
    if plan_id != PLAN_ID:
        raise FailClosed([f"full8-summary plan_id 不匹配: 期望 {PLAN_ID!r}, 实际 {plan_id!r}"])

    problems: list[str] = []
    n_cases = summary.get("n_cases")
    n_selected = summary.get("n_selected")
    n_evaluable = dig(summary, ("aggregate", "n_evaluable_sessions"), "full8-summary")

    if n_cases != REQUIRED_SESSIONS:
        problems.append(f"full8 场次不足: n_cases={n_cases}, 期望 {REQUIRED_SESSIONS}")
    if n_selected != REQUIRED_SESSIONS:
        problems.append(f"full8 未选满: n_selected={n_selected}, 期望 {REQUIRED_SESSIONS}")
    if n_evaluable != REQUIRED_SESSIONS:
        problems.append(f"full8 可评估场次不足: n_evaluable_sessions={n_evaluable}, 期望 {REQUIRED_SESSIONS}")

    per_case = summary.get("per_case_status")
    if not isinstance(per_case, dict):
        problems.append("缺少 per_case_status")
    elif list(per_case.keys()) != list(LOCKED_CASE_IDS):
        # Order AND set must match: a reordered or shortened list is a corruption.
        problems.append(
            "per_case_status 的 case 集不完整/不唯一/顺序不一致，"
            f"期望锁定的 8 场 {list(LOCKED_CASE_IDS)}，实际 {list(per_case.keys())}"
        )
    else:
        for case_id, block in per_case.items():
            status = block.get("status") if isinstance(block, dict) else None
            if not isinstance(status, dict):
                problems.append(f"{case_id}: 缺少 status 块")
                continue
            if not status.get("selected"):
                problems.append(f"{case_id}: 未被选中 (selected=false)")
            if not status.get("evidence_valid"):
                problems.append(f"{case_id}: evidence_valid=false")
            if not status.get("quality_passed"):
                problems.append(f"{case_id}: quality_passed=false")

    gates = summary.get("summary_gates")
    if not isinstance(gates, dict):
        problems.append("缺少 summary_gates")
    else:
        gate_flags = gates.get("gates")
        if not isinstance(gate_flags, dict) or not gate_flags:
            problems.append("summary_gates.gates 缺失或为空")
        else:
            for name, value in sorted(gate_flags.items()):
                if value is not True:
                    problems.append(f"汇总门禁未通过: {name}={value}")
        if gates.get("passed") is not True:
            problems.append(f"summary_gates.passed={gates.get('passed')}")

    if summary.get("errors"):
        problems.append(f"full8-summary 记录了 errors: {summary.get('errors')}")
    if summary.get("integrity_problems"):
        problems.append(f"full8-summary 记录了 integrity_problems: {summary.get('integrity_problems')}")

    if problems:
        raise FailClosed(problems)
    return {
        "n_cases": n_cases,
        "n_selected": n_selected,
        "n_evaluable_sessions": n_evaluable,
        "case_ids": list(per_case.keys()),
        "gates": dig(summary, ("summary_gates", "gates"), "full8-summary"),
    }


def validate_manifest(manifest: dict) -> dict:
    """Reuse the summarizer's strict batch-integrity contract verbatim.

    We do NOT re-implement a weaker parallel validator. Any problem reported by
    ``check_manifest_integrity`` — wrong schema / plan_id, non-empty git status,
    HEAD moved mid-batch, missing or differing runtime source hashes before/after,
    incomplete or out-of-order case set, missing model locks, etc. — fails the
    report closed. This is the single source of truth shared with the continuous-E2E
    driver, so the report can never accept a batch the pipeline itself would reject.
    """
    problems = check_manifest_integrity(manifest)
    if problems:
        raise FailClosed([f"batch-run-manifest 完整性校验失败: {p}" for p in problems])

    git_block = manifest.get("git") or {}
    post = manifest.get("post_run") or {}
    locks = manifest.get("locks") or {}
    model_hashes = (locks.get("model_hashes") or {})
    return {
        "head_sha": git_block.get("head_sha"),
        "enhance_variant": locks.get("enhance_variant"),
        "baseline_variant": locks.get("baseline_variant"),
        "diar_variant": locks.get("diar_variant"),
        "activity_threshold": locks.get("activity_threshold"),
        "asr_model_sha256": model_hashes.get("asr_model_sha256"),
        "sortformer_model_sha256": model_hashes.get("sortformer_model_sha256"),
        "n_runtime_source_hashes": len(git_block.get("runtime_source_sha256") or {}),
    }


def load_case(case_id: str, attempt_dir: Path) -> dict:
    """Read one session's evidence. Any integrity breach is fatal."""
    problems: list[str] = []
    scores = read_json(attempt_dir / CASE_FILES["scores"], f"{case_id}.scores")
    timing = read_json(attempt_dir / CASE_FILES["timing"], f"{case_id}.timing")
    replay = read_json(attempt_dir / CASE_FILES["replay"], f"{case_id}.replay")

    if scores.get("schema") != CASE_SCORE_SCHEMA:
        problems.append(f"{case_id}: scores schema 不匹配 -> {scores.get('schema')!r}")
    if timing.get("schema") != CASE_TIMING_SCHEMA:
        problems.append(f"{case_id}: timing schema 不匹配 -> {timing.get('schema')!r}")
    if scores.get("case_id") != case_id:
        problems.append(f"{case_id}: scores.case_id 不一致 -> {scores.get('case_id')!r}")
    if scores.get("gate_scope") != "full":
        problems.append(f"{case_id}: gate_scope={scores.get('gate_scope')!r}，主对照只接受 full")

    gates = scores.get("gates")
    if not isinstance(gates, dict):
        problems.append(f"{case_id}: 缺少 gates")
        checks: dict = {}
        thresholds: dict = {}
    else:
        checks = gates.get("checks")
        thresholds = gates.get("thresholds")
        if not isinstance(checks, dict):
            problems.append(f"{case_id}: 缺少 gates.checks")
            checks = {}
        if not isinstance(thresholds, dict):
            problems.append(f"{case_id}: 缺少 gates.thresholds")
            thresholds = {}
        # Per-session gates only. DER / overlap recall are aggregate-only by design.
        for name in tuple(EVIDENCE_CHECKS) + tuple(QUALITY_CHECKS):
            if checks.get(name) is not True:
                problems.append(f"{case_id}: 单场门禁未通过 {name}={checks.get(name)!r}")

    if scores.get("asr_model_sha256") != LOCKED_ASR_MODEL_SHA256:
        problems.append(f"{case_id}: ASR 模型 hash 与锁不一致 -> {scores.get('asr_model_sha256')!r}")
    if scores.get("sortformer_model_sha256") != LOCKED_SORTFORMER_MODEL_SHA256:
        problems.append(f"{case_id}: Sortformer 模型 hash 与锁不一致 -> {scores.get('sortformer_model_sha256')!r}")

    # Privacy / integrity gates: these are what make the "isolated replay" claim true.
    for key, expected in (
        ("network_calls", 0),
        ("memory_after", 0),
        ("partial_events", 0),
        ("ui_only_events", 0),
    ):
        value = replay.get(key)
        if value != expected:
            problems.append(f"{case_id}: service-replay {key}={value!r}，期望 {expected}")
    finals = replay.get("transcript_final_events")
    chunks = replay.get("capture_chunk_count")
    if finals != chunks:
        problems.append(f"{case_id}: capture chunk({chunks}) 与 qualified final({finals}) 不相等")
    if finals is None or finals <= 0:
        problems.append(f"{case_id}: 没有 qualified final（finals={finals}），无法作为有效场次")
    if replay.get("audit_contains_embedding") is not False:
        problems.append(f"{case_id}: audit 含 embedding")
    if replay.get("audit_contains_pcm") is not False:
        problems.append(f"{case_id}: audit 含 PCM")

    if problems:
        raise FailClosed(problems)

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        block = dig(scores, ("variants", variant), f"{case_id}.scores")
        metrics[variant] = {}
        for metric, (path, num_key, den_key) in METRIC_SOURCES.items():
            src = dig(block, path, f"{case_id}.{variant}")
            numerator = as_number(dig(src, (num_key,), f"{case_id}.{variant}.{metric}"), f"{case_id}.{variant}.{metric}.{num_key}")
            denominator = as_number(dig(src, (den_key,), f"{case_id}.{variant}.{metric}"), f"{case_id}.{variant}.{metric}.{den_key}")
            metrics[variant][metric] = {
                "numerator": numerator,
                "denominator": denominator,
                "rate": ratio(numerator, denominator, f"{case_id}.{variant}.{metric}"),
            }

    # DER / overlap recall must be identical across variants: that identity IS the
    # evidence that they come from one shared diarization track.
    for metric in SHARED_GATE_METRICS:
        left = metrics[ENHANCED_VARIANT][metric]
        right = metrics[BASELINE_VARIANT][metric]
        if left["numerator"] != right["numerator"] or left["denominator"] != right["denominator"]:
            raise FailClosed(
                [f"{case_id}: {metric} 在 {ENHANCED_VARIANT}/{BASELINE_VARIANT} 不一致，不能作为共享分音门禁"]
            )

    processed = as_number(timing.get("processed_seconds"), f"{case_id}.processed_seconds")
    process_seconds = as_number(timing.get("full_process_seconds"), f"{case_id}.full_process_seconds")
    if processed <= 0 or process_seconds <= 0:
        raise FailClosed([f"{case_id}: 时长数据无效 (processed={processed}, process_seconds={process_seconds})"])
    rtf_max = dig(scores, ("gates", "thresholds", "rtf_max"), f"{case_id}.gates")

    return {
        "case_id": case_id,
        "attempt_dir": str(attempt_dir),
        "metrics": metrics,
        "processed_seconds": processed,
        "process_seconds": process_seconds,
        "rtf_max": float(rtf_max),
        "thresholds": {
            "der_max": as_number(thresholds.get("der_max"), f"{case_id}.gates.thresholds.der_max"),
            "overlap_recall_min": as_number(
                thresholds.get("overlap_recall_min"), f"{case_id}.gates.thresholds.overlap_recall_min"
            ),
            "rtf_max": float(rtf_max),
        },
        # Recorded for disclosure: these two are aggregate-only gates, so a False
        # here does NOT fail the session (see summarize_continuous_full8).
        "aggregate_only_checks": {name: checks.get(name) for name in AGGREGATE_ONLY_CHECKS},
        "scorer_local_passed": gates.get("passed") if isinstance(gates, dict) else None,
        "replay": {
            "qualified_finals": finals,
            "capture_chunks": chunks,
            "speech_rejected_events": replay.get("speech_rejected_events"),
            "partial_events": replay.get("partial_events"),
            "ui_only_events": replay.get("ui_only_events"),
            "network_calls": replay.get("network_calls"),
            "memory_after": replay.get("memory_after"),
            "timeline_entries_added": replay.get("timeline_entries_added"),
            "capture_status_after_finish": replay.get("capture_status_after_finish"),
            "audit_bytes": replay.get("audit_bytes"),
            "audit_contains_embedding": replay.get("audit_contains_embedding"),
            "audit_contains_pcm": replay.get("audit_contains_pcm"),
        },
        "asr_model_sha256": scores.get("asr_model_sha256"),
        "sortformer_model_sha256": scores.get("sortformer_model_sha256"),
        "activity_threshold": scores.get("activity_threshold"),
    }


def load_cases(cfg: Config, case_ids: list[str], manifest: dict) -> list[dict]:
    """Resolve every selected attempt through the summarizer's escape-proof helper.

    ``resolve_attempt_dir`` rejects ``../`` traversal, absolute paths outside the batch
    directory, and symlink escapes (it resolves the symlink before the boundary check),
    so we can only ever read this batch's own selected attempt directories.
    """
    attempt_dirs: dict[str, Path] = {}
    for case_id in case_ids:
        sel = dig(manifest, ("attempts", case_id, "selected_attempt_dir"), "manifest")
        resolved, err = resolve_attempt_dir(cfg.batch_dir, sel)
        if err:
            raise FailClosed([f"{case_id}: selected_attempt_dir 解析被拒: {err}"])
        attempt_dirs[case_id] = resolved

    cases = [load_case(case_id, attempt_dirs[case_id]) for case_id in case_ids]

    rtf_limits = {case["rtf_max"] for case in cases}
    if len(rtf_limits) != 1:
        raise FailClosed([f"各场次 RTF 阈值不一致: {sorted(rtf_limits)}"])
    asr_hashes = {case["asr_model_sha256"] for case in cases}
    if len(asr_hashes) != 1 or not all(asr_hashes):
        raise FailClosed([f"各场次 ASR 模型 hash 不一致或缺失: {sorted(asr_hashes)}"])
    sf_hashes = {case["sortformer_model_sha256"] for case in cases}
    if len(sf_hashes) != 1 or not all(sf_hashes):
        raise FailClosed([f"各场次 Sortformer 模型 hash 不一致或缺失: {sorted(sf_hashes)}"])
    thresholds = {case["activity_threshold"] for case in cases}
    if len(thresholds) != 1:
        raise FailClosed([f"各场次 activity_threshold 不一致: {sorted(thresholds)}"])
    return cases


# --------------------------------------------------------------------------- #
# Aggregation (weighted, never an average of percentages)
# --------------------------------------------------------------------------- #
def aggregate_metric(cases: list[dict], variant: str, metric: str) -> dict:
    numerator = sum(case["metrics"][variant][metric]["numerator"] for case in cases)
    denominator = sum(case["metrics"][variant][metric]["denominator"] for case in cases)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": ratio(numerator, denominator, f"aggregate.{variant}.{metric}"),
        "n_cases": len(cases),
        "aggregation": "sum(numerator)/sum(denominator)，不是逐场百分比平均",
    }


def build_main_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for metric in MAIN_METRICS:
        baseline = aggregate_metric(cases, BASELINE_VARIANT, metric)
        enhanced = aggregate_metric(cases, ENHANCED_VARIANT, metric)
        if baseline["denominator"] != enhanced["denominator"]:
            raise FailClosed(
                [
                    f"{metric}: {BASELINE_VARIANT}/{ENHANCED_VARIANT} 分母不同 "
                    f"({baseline['denominator']} vs {enhanced['denominator']})，不能做同口径对照"
                ]
            )
        delta_rate = baseline["rate"] - enhanced["rate"]
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "baseline_variant": BASELINE_VARIANT,
                "enhanced_variant": ENHANCED_VARIANT,
                "baseline": baseline,
                "enhanced": enhanced,
                "delta_rate": delta_rate,
                "delta_percentage_points": pct(delta_rate),
                "relative_error_reduction_pct": pct(delta_rate / baseline["rate"]),
                "comparable": True,
                "evidence_boundary": "同一 full8 批次、同 8 场、同分音轨道、同 ASR、同金标；唯一有意变化是空间增强通道配置。",
            }
        )
    return rows


def cross_check_summary(summary: dict, cases: list[dict]) -> dict:
    """Recomputed aggregates must reproduce the published summary exactly."""
    problems: list[str] = []
    detail: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        for metric in MAIN_METRICS + SHARED_GATE_METRICS:
            recomputed = aggregate_metric(cases, variant, metric)
            block = dig(summary, ("aggregate", "variants", variant, metric), "full8-summary")
            _, num_key, den_key = METRIC_SOURCES[metric]
            published_num = as_number(block.get(num_key), f"summary.{variant}.{metric}.{num_key}")
            published_den = as_number(block.get(den_key), f"summary.{variant}.{metric}.{den_key}")
            published_rate = as_number(block.get("rate"), f"summary.{variant}.{metric}.rate")
            if block.get("aggregate_denominator_zero") is not False:
                problems.append(f"summary.{variant}.{metric}.aggregate_denominator_zero 标记异常")
            if recomputed["numerator"] != published_num or recomputed["denominator"] != published_den:
                problems.append(
                    f"summary.{variant}.{metric} 分子/分母不一致: 重算 "
                    f"{recomputed['numerator']}/{recomputed['denominator']} vs 发布 {published_num}/{published_den}"
                )
            if abs(recomputed["rate"] - published_rate) > 1e-9:
                problems.append(
                    f"summary.{variant}.{metric} 比率不一致: 重算 {recomputed['rate']!r} vs 发布 {published_rate!r}"
                )
            detail[f"{variant}.{metric}"] = {
                "recomputed": recomputed,
                "published": {"numerator": published_num, "denominator": published_den, "rate": published_rate},
                "match": (
                    recomputed["numerator"] == published_num
                    and recomputed["denominator"] == published_den
                    and abs(recomputed["rate"] - published_rate) <= 1e-9
                ),
            }
    if problems:
        raise FailClosed(problems)
    return detail


def cross_check_sortformer(sortformer: Optional[dict], cases: list[dict], diar_variant: Optional[str]) -> Optional[dict]:
    if sortformer is None:
        return None
    if sortformer.get("schema") != SORTFORMER_SCORE_SCHEMA:
        raise FailClosed([f"sortformer scores schema 不匹配 -> {sortformer.get('schema')!r}"])
    variant = diar_variant or "micA_mean"
    block = dig(sortformer, ("variants", variant, "aggregate"), "sortformer-scores")
    recomputed_der = aggregate_metric(cases, ENHANCED_VARIANT, "der")
    recomputed_recall = aggregate_metric(cases, ENHANCED_VARIANT, "recall")
    expected = {
        "der": (
            as_number(block.get("errors"), "sortformer.der.errors"),
            as_number(block.get("reference_speaker_frames"), "sortformer.der.reference_speaker_frames"),
        ),
        "recall": (
            as_number(block.get("overlap_detected_frames"), "sortformer.recall.overlap_detected_frames"),
            as_number(block.get("overlap_reference_frames"), "sortformer.recall.overlap_reference_frames"),
        ),
    }
    actual = {
        "der": (recomputed_der["numerator"], recomputed_der["denominator"]),
        "recall": (recomputed_recall["numerator"], recomputed_recall["denominator"]),
    }
    if expected != actual:
        raise FailClosed([f"sortformer {variant} 分音聚合与逐场重算不一致: {actual} vs {expected}"])
    return {
        "variant": variant,
        "der": {"numerator": actual["der"][0], "denominator": actual["der"][1], "rate": recomputed_der["rate"]},
        "recall": {"numerator": actual["recall"][0], "denominator": actual["recall"][1], "rate": recomputed_recall["rate"]},
        "match": True,
    }


def build_shared_gate(cases: list[dict], sortformer_check: Optional[dict]) -> dict:
    der = aggregate_metric(cases, ENHANCED_VARIANT, "der")
    recall = aggregate_metric(cases, ENHANCED_VARIANT, "recall")
    return {
        "shared_across_variants": True,
        "shared_source": "同一 Sortformer 匿名轨道（micA_mean），all8 与 ch0 共用",
        "attribution_allowed": False,
        "attribution_rule": "DER 与 overlap recall 是分音前置门禁：all8 与 ch0 数值相同，不能写成 all8 改善了 DER。",
        "der": {
            "numerator": der["numerator"],
            "denominator": der["denominator"],
            "rate": der["rate"],
            "label": METRIC_LABELS["der"],
        },
        "overlap_recall": {
            "numerator": recall["numerator"],
            "denominator": recall["denominator"],
            "rate": recall["rate"],
            "label": METRIC_LABELS["recall"],
        },
        "sortformer_cross_check": sortformer_check,
    }


def build_aggregate_gates(cases: list[dict], shared_gate: dict) -> dict:
    """Evaluate the pooled DER / overlap-recall gates plus the disclosure list.

    Both are aggregate-only gates in this pipeline: a single session below the
    recall threshold is disclosed, never treated as a session failure.
    """
    der_max = cases[0]["thresholds"]["der_max"]
    recall_min = cases[0]["thresholds"]["overlap_recall_min"]
    for case in cases:
        if case["thresholds"]["der_max"] != der_max or case["thresholds"]["overlap_recall_min"] != recall_min:
            raise FailClosed([f"{case['case_id']}: 门禁阈值与其他场次不一致"])

    checks = {
        "aggregate_der_le_max": shared_gate["der"]["rate"] <= der_max,
        "aggregate_overlap_recall_ge_min": shared_gate["overlap_recall"]["rate"] >= recall_min,
    }
    below = sorted(
        (
            {
                "case_id": case["case_id"],
                "overlap_recall": case["metrics"][ENHANCED_VARIANT]["recall"]["rate"],
                "numerator": case["metrics"][ENHANCED_VARIANT]["recall"]["numerator"],
                "denominator": case["metrics"][ENHANCED_VARIANT]["recall"]["denominator"],
            }
            for case in cases
            if case["metrics"][ENHANCED_VARIANT]["recall"]["rate"] < recall_min
        ),
        key=lambda item: item["overlap_recall"],
    )
    return {
        "thresholds": {"der_max": der_max, "overlap_recall_min": recall_min},
        "checks": checks,
        "passed": all(checks.values()),
        "sessions_below_recall_threshold": below,
        "note": (
            "DER 与 overlap recall 在本管线中定义为聚合门禁（对 8 场 pooled 分子/分母求值），"
            "单场不设门禁；单场低于阈值只作披露，不判该场失败。"
        ),
    }


def build_rtf(cases: list[dict]) -> dict:
    total_process = sum(case["process_seconds"] for case in cases)
    total_audio = sum(case["processed_seconds"] for case in cases)
    rtf = ratio(total_process, total_audio, "aggregate.rtf")
    rtf_max = cases[0]["rtf_max"]
    return {
        "basis": "all8 全链外层 wrapper 加权：sum(full_process_seconds) / sum(processed_seconds)",
        "total_process_seconds": total_process,
        "total_audio_seconds": total_audio,
        "rtf": rtf,
        "rtf_max": rtf_max,
        "passed": rtf <= rtf_max,
        "not_a_speed_comparison": True,
        "note": "本次只测了 all8 全链闭环耗时，没有同口径的 ch0 全链耗时，因此 RTF 不参与前后收益对照。",
    }


def build_capture_replay(cases: list[dict]) -> dict:
    keys = (
        "qualified_finals",
        "capture_chunks",
        "speech_rejected_events",
        "partial_events",
        "ui_only_events",
        "network_calls",
        "memory_after",
        "timeline_entries_added",
        "audit_bytes",
    )
    totals = {key: sum(int(case["replay"][key] or 0) for case in cases) for key in keys}
    statuses = sorted({str(case["replay"]["capture_status_after_finish"]) for case in cases})
    chunk_matches_final = all(
        case["replay"]["capture_chunks"] == case["replay"]["qualified_finals"] for case in cases
    )
    return {
        "totals": totals,
        "capture_status_after_finish": statuses,
        "chunk_equals_final_per_case": chunk_matches_final,
        "audit_contains_embedding": any(case["replay"]["audit_contains_embedding"] for case in cases),
        "audit_contains_pcm": any(case["replay"]["audit_contains_pcm"] for case in cases),
        "evidence_boundary": (
            "隔离 capture 回放证据：qualified final 与 capture chunk 一一对应只证明回放链路一致，"
            "timeline_entries_added=0、capture 以 interrupted 结束，因此不等于长期归档 / 长期 Timeline 写入成功。"
        ),
    }


def rank_cases(cases: list[dict], variant: str, metric: str) -> list[dict]:
    """Worst first. Ranking uses the recomputed rate of each session."""
    ranked = sorted(
        cases,
        key=lambda case: (-case["metrics"][variant][metric]["rate"], case["case_id"]),
    )
    return [
        {
            "case_id": case["case_id"],
            "rank": index + 1,
            "numerator": case["metrics"][variant][metric]["numerator"],
            "denominator": case["metrics"][variant][metric]["denominator"],
            "rate": case["metrics"][variant][metric]["rate"],
        }
        for index, case in enumerate(ranked)
    ]


def build_worst_case(cases: list[dict], summary: dict) -> dict:
    rankings = {metric: rank_cases(cases, ENHANCED_VARIANT, metric) for metric in MAIN_METRICS}
    worst_id = rankings["cpcer"][0]["case_id"]
    worst_case = next(case for case in cases if case["case_id"] == worst_id)

    # The summary publishes its own worst ranking; it must agree with our recomputation.
    problems = []
    for metric, key in (("cpcer", "cpcer"), ("overlap", "overlap_cer"), ("non_overlap", "non_overlap_cer")):
        published = dig(summary, ("worst_case_ranking", key, "worst"), "full8-summary")
        if not isinstance(published, list) or not published:
            problems.append(f"summary.worst_case_ranking.{key}.worst 缺失")
            continue
        if published[0] != rankings[metric][0]["case_id"]:
            problems.append(
                f"summary.worst_case_ranking.{key} 最差场不一致: 重算 {rankings[metric][0]['case_id']} vs 发布 {published[0]}"
            )
    if problems:
        raise FailClosed(problems)

    return {
        "case_id": worst_id,
        "selection_basis": f"按 {ENHANCED_VARIANT} 的 cpCER 从差到好排序的第一名",
        "metrics": {
            metric: {
                "numerator": worst_case["metrics"][ENHANCED_VARIANT][metric]["numerator"],
                "denominator": worst_case["metrics"][ENHANCED_VARIANT][metric]["denominator"],
                "rate": worst_case["metrics"][ENHANCED_VARIANT][metric]["rate"],
            }
            for metric in MAIN_METRICS
        },
        "rankings": rankings,
        "note": "八场汇总通过不代表每场都好；最差场必须显式列出，不能被聚合平均掩盖。",
    }


# --------------------------------------------------------------------------- #
# Milestones (three tiers, never cross-compared)
# --------------------------------------------------------------------------- #
def assert_no_cross_milestone_delta(milestones: list[dict]) -> None:
    """Hard guard: milestone rows must never carry a delta or a comparability flag."""
    problems = []
    for row in milestones:
        if row.get("comparable_to_main_table"):
            problems.append(f"里程碑 {row.get('id')} 被标记为可与主表比较")
        if row.get("delta_vs_main") is not None:
            problems.append(f"里程碑 {row.get('id')} 生成了跨越证据边界的 delta")
        if "evidence_boundary" not in row or not row["evidence_boundary"]:
            problems.append(f"里程碑 {row.get('id')} 缺少证据边界说明")
    if problems:
        raise FailClosed(problems)


def build_oracle_milestone(cfg: Config) -> dict:
    if cfg.oracle_json is None:
        return {
            "id": "oracle_mvdr",
            "tier": "oracle_upper_bound",
            "label": "Oracle MVDR 上限",
            "source_kind": "plan_note",
            "source": {"path": None, "note": cfg.oracle_note},
            "metrics": None,
            "comparable_to_main_table": False,
            "delta_vs_main": None,
            "evidence_boundary": "人工 RTTM 掩码的研发上限，非自动链路产出，也没有机器可读源文件。",
            "cannot_prove": [
                "不能证明自动链路达到该上限",
                "不能与 full8 主表做百分点差或相对提升计算",
                "不能写成已部署效果",
            ],
        }

    note = read_json(cfg.oracle_json, "oracle-json")
    if note.get("schema") != ORACLE_NOTE_SCHEMA:
        raise FailClosed([f"oracle-json schema 不匹配: 期望 {ORACLE_NOTE_SCHEMA}, 实际 {note.get('schema')!r}"])
    return {
        "id": "oracle_mvdr",
        "tier": "oracle_upper_bound",
        "label": "Oracle MVDR 上限",
        "source_kind": "machine_readable",
        "source": {"path": str(cfg.oracle_json), "sha256": sha256_file(cfg.oracle_json), "note": note.get("note", "")},
        "metrics": note.get("metrics"),
        "comparable_to_main_table": False,
        "delta_vs_main": None,
        "evidence_boundary": str(note.get("evidence_boundary", "人工 RTTM 掩码的研发上限，不与连续自动链路同口径。")),
        "cannot_prove": list(
            note.get("cannot_prove", ["不能与 full8 主表做百分点差或相对提升计算", "不能写成已部署效果"])
        ),
    }


def build_auto_mask_milestone(retention: dict, path: Path) -> dict:
    if retention.get("schema") != RETENTION_SCHEMA:
        raise FailClosed([f"auto-mask retention schema 不匹配 -> {retention.get('schema')!r}"])
    automatic = dig(retention, ("automatic",), "retention")
    baseline = dig(retention, ("baseline",), "retention")
    scope = dig(retention, ("scope",), "retention")
    if retention.get("passed") is not True:
        raise FailClosed([f"auto-mask retention.passed={retention.get('passed')}"])
    return {
        "id": "auto_mask_component",
        "tier": "automatic_component",
        "label": "自动掩码多通道组件",
        "source_kind": "machine_readable",
        "source": {"path": str(path), "sha256": sha256_file(path)},
        "metrics": {
            "overlap_cer_baseline": baseline.get("overlap_cer"),
            "overlap_cer_automatic": automatic.get("overlap_cer"),
            "non_overlap_cer_automatic": automatic.get("non_overlap_cer"),
            "frontend_rtf": automatic.get("frontend_rtf"),
            "cases_improved_vs_ch0": automatic.get("cases_improved_vs_ch0"),
            "enhance_variant": retention.get("enhance_variant"),
            "activity_threshold": retention.get("activity_threshold"),
        },
        "comparable_to_main_table": False,
        "delta_vs_main": None,
        "evidence_boundary": (
            "组件级验证：使用金标片段边界与评分子映射，且 continuous_asr="
            f"{scope.get('continuous_asr')}，不是连续会议全链输出，"
            "因此不能与 full8 主表做同口径百分点比较。"
        ),
        "cannot_prove": [
            "不能证明连续会议全链路达到该数值",
            "不能与 full8 主表相减或计算相对提升",
            "不能替代 8 场连续 E2E 的部署结论",
        ],
    }


def build_full8_milestone(main_rows: list[dict], rtf: dict, n_sessions: int, total_hours: float) -> dict:
    return {
        "id": "continuous_full8_e2e",
        "tier": "continuous_e2e",
        "label": "连续 full8 E2E（主对照所在层）",
        "source_kind": "machine_readable",
        "source": {"path": None, "note": "见 comparison.json.inputs 中的 full8 输入"},
        "metrics": {
            row["metric"]: {
                "ch0_rate": row["baseline"]["rate"],
                "all8_rate": row["enhanced"]["rate"],
                "delta_percentage_points": row["delta_percentage_points"],
                "relative_error_reduction_pct": row["relative_error_reduction_pct"],
            }
            for row in main_rows
        },
        "sessions": n_sessions,
        "total_audio_hours": total_hours,
        "full_loop_rtf": rtf["rtf"],
        "comparable_to_main_table": False,
        "delta_vs_main": None,
        "evidence_boundary": "主表本身所在的证据层；与主表同源，不再额外计算“相对主表”的提升。",
        "cannot_prove": [
            "不能外推到 AliMeeting 之外的设备几何或真实佩戴噪声",
            "不能等同于 Android / 眼镜真机验收",
        ],
    }


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def build_comparison(cfg: Config) -> dict:
    summary = read_json(cfg.full8_summary, "full8-summary")
    manifest = read_json(cfg.batch_manifest, "batch-manifest")
    retention = read_json(cfg.auto_mask_retention, "auto-mask-retention")

    summary_info = validate_summary(summary)
    manifest_info = validate_manifest(manifest)
    cases = load_cases(cfg, summary_info["case_ids"], manifest)

    recompute_check = cross_check_summary(summary, cases)

    sortformer = None
    sortformer_check = None
    if cfg.sortformer_scores is not None:
        sortformer = read_json(cfg.sortformer_scores, "sortformer-scores")
        sortformer_check = cross_check_sortformer(sortformer, cases, manifest_info.get("diar_variant"))

    main_rows = build_main_rows(cases)
    shared_gate = build_shared_gate(cases, sortformer_check)
    aggregate_gates = build_aggregate_gates(cases, shared_gate)
    if not aggregate_gates["passed"]:
        failed = [name for name, ok in aggregate_gates["checks"].items() if not ok]
        raise FailClosed([f"聚合门禁未通过: {', '.join(failed)}"])
    rtf = build_rtf(cases)
    if not rtf["passed"]:
        raise FailClosed([f"all8 全链 RTF 超阈值: {rtf['rtf']} > {rtf['rtf_max']}"])
    capture_replay = build_capture_replay(cases)
    worst_case = build_worst_case(cases, summary)

    milestones = [
        build_oracle_milestone(cfg),
        build_auto_mask_milestone(retention, cfg.auto_mask_retention),
        build_full8_milestone(
            main_rows,
            rtf,
            len(cases),
            rtf["total_audio_seconds"] / 3600.0,
        ),
    ]
    assert_no_cross_milestone_delta(milestones)

    # Absolute paths: the report must be reproducible from another working directory.
    inputs = [
        {"role": "full8_summary", "path": str(cfg.full8_summary.resolve()), "sha256": sha256_file(cfg.full8_summary), "schema": summary.get("schema")},
        {"role": "batch_manifest", "path": str(cfg.batch_manifest.resolve()), "sha256": sha256_file(cfg.batch_manifest), "schema": manifest.get("schema")},
        {"role": "batch_dir", "path": str(cfg.batch_dir.resolve()), "sha256": None, "schema": "directory"},
        {"role": "auto_mask_retention", "path": str(cfg.auto_mask_retention.resolve()), "sha256": sha256_file(cfg.auto_mask_retention), "schema": retention.get("schema")},
    ]
    if cfg.sortformer_scores is not None and sortformer is not None:
        inputs.append(
            {
                "role": "sortformer_scores",
                "path": str(cfg.sortformer_scores),
                "sha256": sha256_file(cfg.sortformer_scores),
                "schema": sortformer.get("schema"),
            }
        )
    if cfg.oracle_json is not None:
        oracle_row = next(row for row in milestones if row["id"] == "oracle_mvdr")
        inputs.append(
            {
                "role": "oracle_milestone",
                "path": str(cfg.oracle_json),
                "sha256": oracle_row["source"].get("sha256"),
                "schema": ORACLE_NOTE_SCHEMA,
            }
        )

    git_state = gitignore_state(cfg.out_dir)
    per_case = {
        case["case_id"]: {
            variant: {
                metric: {
                    "numerator": case["metrics"][variant][metric]["numerator"],
                    "denominator": case["metrics"][variant][metric]["denominator"],
                    "rate": case["metrics"][variant][metric]["rate"],
                }
                for metric in MAIN_METRICS
            }
            for variant in VARIANTS
        }
        for case in cases
    }

    return {
        "schema": SCHEMA_OUT,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "AliMeeting（公开多通道会议数据集，离线软件链路）",
        "claim_scope": (
            "结论仅限 AliMeeting 公开数据上的离线软件链路证据；"
            "不等于眼镜阵列几何、不等于真机收音、不等于 Android 或眼镜硬件验收。"
        ),
        "headline_claim": "多通道空间前端（all8 MVDR）改善了同一 ASR 的输入质量，从而降低端到端 CER。",
        "inputs": inputs,
        "out_dir": str(cfg.out_dir.resolve()),
        "oracle_note": cfg.oracle_note,
        "provenance": {
            "head_sha": manifest_info["head_sha"],
            "n_runtime_source_hashes": manifest_info["n_runtime_source_hashes"],
            "asr_model_sha256": manifest_info["asr_model_sha256"],
            "sortformer_model_sha256": manifest_info["sortformer_model_sha256"],
            "enhance_variant": manifest_info["enhance_variant"],
            "baseline_variant": manifest_info["baseline_variant"],
            "diar_variant": manifest_info["diar_variant"],
            "activity_threshold": manifest_info["activity_threshold"],
        },
        "coverage": {
            "expected_sessions": REQUIRED_SESSIONS,
            "n_cases": summary_info["n_cases"],
            "n_selected": summary_info["n_selected"],
            "n_evaluable_sessions": summary_info["n_evaluable_sessions"],
            "case_ids": summary_info["case_ids"],
            "total_audio_seconds": rtf["total_audio_seconds"],
            "total_audio_hours": rtf["total_audio_seconds"] / 3600.0,
        },
        "main_table": main_rows,
        "shared_diarization_gate": shared_gate,
        "aggregate_gates": aggregate_gates,
        "rtf": rtf,
        "capture_replay": capture_replay,
        "worst_case": worst_case,
        "milestones": milestones,
        "summary_gates": summary_info["gates"],
        "recompute_cross_check": recompute_check,
        "output_gitignore": git_state,
        "gates": {
            "inputs_present": True,
            "schemas_expected": True,
            "sessions_complete": True,
            "denominators_nonzero": True,
            "summary_gates_passed": True,
            "per_case_gates_passed": True,
            "recompute_matches_published": True,
            "no_cross_milestone_delta": True,
            "rtf_within_threshold": True,
            "aggregate_gates_passed": True,
            "passed": True,
        },
        # Rendering-only per-case index; stripped before comparison.json is written.
        "_per_case": per_case,
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _csv_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for row in data["main_table"]:
        variant_label = {"ch0": "ch0（单通道基线）", "all8": "all8（多通道空间前端）"}
        rows.append(
            {
                "scope": "main",
                "group": row["metric"],
                "subject": row["baseline_variant"],
                "numerator": row["baseline"]["numerator"],
                "denominator": row["baseline"]["denominator"],
                "rate_pct": f"{pct(row['baseline']['rate']):.4f}",
                "delta_percentage_points": "",
                "relative_error_reduction_pct": "",
                "note": variant_label.get(row["baseline_variant"], row["baseline_variant"]),
            }
        )
        rows.append(
            {
                "scope": "main",
                "group": row["metric"],
                "subject": row["enhanced_variant"],
                "numerator": row["enhanced"]["numerator"],
                "denominator": row["enhanced"]["denominator"],
                "rate_pct": f"{pct(row['enhanced']['rate']):.4f}",
                "delta_percentage_points": f"{row['delta_percentage_points']:.4f}",
                "relative_error_reduction_pct": f"{row['relative_error_reduction_pct']:.4f}",
                "note": variant_label.get(row["enhanced_variant"], row["enhanced_variant"]),
            }
        )

    gate = data["shared_diarization_gate"]
    for metric, block in (("der", gate["der"]), ("overlap_recall", gate["overlap_recall"])):
        rows.append(
            {
                "scope": "shared_gate",
                "group": metric,
                "subject": "micA_mean（共享分音轨道）",
                "numerator": block["numerator"],
                "denominator": block["denominator"],
                "rate_pct": f"{pct(block['rate']):.4f}",
                "delta_percentage_points": "",
                "relative_error_reduction_pct": "",
                "note": "共享分音门禁，禁止归因给 all8",
            }
        )

    for case_id, per_case in sorted(data["_per_case"].items()):
        for variant in VARIANTS:
            for metric in MAIN_METRICS:
                block = per_case[variant][metric]
                rows.append(
                    {
                        "scope": "per_case",
                        "group": metric,
                        "subject": f"{case_id}:{variant}",
                        "numerator": block["numerator"],
                        "denominator": block["denominator"],
                        "rate_pct": f"{pct(block['rate']):.4f}",
                        "delta_percentage_points": "",
                        "relative_error_reduction_pct": "",
                        "note": "逐场明细（供复核，不参与聚合平均）",
                    }
                )
    return rows


def write_metrics_csv(data: dict, path: Path) -> None:
    fields = [
        "scope",
        "group",
        "subject",
        "numerator",
        "denominator",
        "rate_pct",
        "delta_percentage_points",
        "relative_error_reduction_pct",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _csv_rows(data):
            writer.writerow(row)


def render_report(data: dict) -> str:
    main_rows = data["main_table"]
    gate = data["shared_diarization_gate"]
    agg = data["aggregate_gates"]
    rtf = data["rtf"]
    replay = data["capture_replay"]["totals"]
    worst = data["worst_case"]
    coverage = data["coverage"]
    provenance = data["provenance"]

    lines: list[str] = []
    add = lines.append

    add("# AliMeeting 多通道音频前端优化前后对比")
    add("")
    add(f"- 生成时间（UTC）：{data['generated_at']}")
    add(f"- 数据集：{data['dataset']}")
    add(f"- 覆盖：{coverage['n_selected']}/{coverage['expected_sessions']} 场，"
        f"共 {coverage['total_audio_seconds']:.3f} 秒（约 {coverage['total_audio_hours']:.2f} 小时）")
    add(f"- 代码锁：head `{provenance['head_sha']}`，锁定源文件 {provenance['n_runtime_source_hashes']} 个")
    add(f"- 机器可读数据：`comparison.json`；逐场与聚合明细：`metrics.csv`")
    add("")

    # ---- 1. 一句话结论 ----
    by_metric = {row["metric"]: row for row in main_rows}
    cpcer = by_metric["cpcer"]
    add("## 1. 一句话结论")
    add("")
    add(f"在 AliMeeting 公开多通道会议数据上，把空间增强从单通道 {BASELINE_VARIANT} 换成 8 通道 all8 后，"
        f"**同一个 ASR 模型**的连续识别错误显著下降：cpCER 从 {fmt_pct(cpcer['baseline']['rate'])} 降到 "
        f"{fmt_pct(cpcer['enhanced']['rate'])}，相对错误降低 {cpcer['relative_error_reduction_pct']:.2f}%。")
    add("")
    add(f"该结论建立在 {coverage['n_selected']} 场、约 {coverage['total_audio_hours']:.2f} 小时的 AliMeeting 离线软件链路上；"
        "**这是公开数据集上的软件证据，不是眼镜阵列或真机验收**。")
    add("")
    add(f"归因边界：{data['headline_claim']}")
    add("")

    # ---- 2. 核心前后对比 ----
    add("## 2. 核心前后对比（同一 full8 批次内 ch0 vs all8）")
    add("")
    add("| 指标 | ch0（单通道基线） | all8（多通道空间前端） | 百分点变化（ch0 − all8） | 相对错误降低 |")
    add("|---|---|---|---|---|")
    for row in main_rows:
        add(
            f"| {row['label']} | {fmt_pct(row['baseline']['rate'])} "
            f"({int(row['baseline']['numerator'])}/{int(row['baseline']['denominator'])}) | "
            f"{fmt_pct(row['enhanced']['rate'])} "
            f"({int(row['enhanced']['numerator'])}/{int(row['enhanced']['denominator'])}) | "
            f"{row['delta_percentage_points']:+.2f} pp | {row['relative_error_reduction_pct']:.2f}% |"
        )
    add("")
    add("读法说明：")
    add("")
    add("- 所有比率都由 8 场的原始分子/分母求和后再相除，不是逐场百分比取平均。")
    add(f"- 两个变体共用同一分母，唯一有意变化是空间增强通道配置；ASR 模型 `{provenance['asr_model_sha256'][:12]}…` 前后完全一致。")
    add("- 因此这里证明的是「多通道空间前端提供给同一 ASR 的输入更干净」，**不是**换用或训练了更强的 ASR 模型。")
    add("")

    # ---- 3. 已验证什么 ----
    add("## 3. 这条链路已验证什么")
    add("")
    add("| 项目 | 数值 | 它证明了什么 |")
    add("|---|---|---|")
    add(
        f"| 共享分音门禁 · DER | {fmt_pct(gate['der']['rate'])} "
        f"({int(gate['der']['numerator'])}/{int(gate['der']['denominator'])}) | "
        "分音前置门禁通过；all8 与 ch0 共用同一 Sortformer 匿名轨道，数值相同，"
        "**不能写成 all8 改善了 DER**。 |"
    )
    add(
        f"| 共享分音门禁 · overlap recall | {fmt_pct(gate['overlap_recall']['rate'])} "
        f"({int(gate['overlap_recall']['numerator'])}/{int(gate['overlap_recall']['denominator'])}) | "
        "同上，属共享分音能力，不归因给增强变体。 |"
    )
    add(
        f"| all8 全链 RTF（外层加权） | {rtf['rtf']:.3f}（阈值 ≤ {rtf['rtf_max']:.1f}） | "
        "all8 全链可在实时预算内跑完；**这不是 ch0 与 all8 的速度对比**，本次没有同口径 ch0 全链耗时。 |"
    )
    add(
        f"| 聚合门禁 · DER ≤ {agg['thresholds']['der_max']:.2f} / overlap recall ≥ "
        f"{agg['thresholds']['overlap_recall_min']:.2f} | "
        f"pooled DER {fmt_pct(gate['der']['rate'])}、pooled recall {fmt_pct(gate['overlap_recall']['rate'])}，均通过 | "
        "这两个指标按管线定义只在 8 场 pooled 分子/分母上判定，单场不设门禁。 |"
    )
    add(f"| 8/8 汇总门禁 | 全部通过 | {coverage['n_selected']}/{coverage['expected_sessions']} 场全部 selected、evidence_valid、quality_passed。 |")
    add(
        f"| 逐场 result + evidence 门禁 | {coverage['n_selected']}/{coverage['expected_sessions']} 通过 | "
        "每场的隐私/完整性门禁与结果门禁（final 非空、cpCER 与 overlap CER 严格优于 ch0、内部 RTF）均通过。 |"
    )
    add(
        f"| 隔离 capture 回放 | {replay['qualified_finals']} qualified final / {replay['capture_chunks']} capture chunk，"
        f"{replay['speech_rejected_events']} rejected | "
        "逐场 final 与 chunk 一一对应，回放链路一致。 |"
    )
    add(
        f"| 隐私与隔离计数 | 网络调用 {replay['network_calls']}、长期记忆写入 {replay['memory_after']}、"
        f"partial 持久化 {replay['partial_events']}、UI-only 持久化 {replay['ui_only_events']} | "
        "回放过程零网络、零长期记忆、零 partial/UI-only 持久化；audit 无 embedding、无 PCM。 |"
    )
    add("")
    add("⚠️ capture 回放边界：本项只证明「隔离 capture 回放一致」，"
        f"timeline_entries_added={replay['timeline_entries_added']}、capture 以 "
        f"{'/'.join(data['capture_replay']['capture_status_after_finish'])} 结束，**不等于长期归档或长期 Timeline 写入成功**。")
    add("")

    # ---- 4. 还没有证明什么 ----
    add("## 4. 还没有证明什么 / 当前风险")
    add("")
    add("- **AliMeeting ≠ 眼镜阵列或真机收音**：公开会议数据的通道排列、佩戴噪声、麦克风位置与真实日常场景都不同，"
        "本结论不能外推成硬件或真机效果。")
    add("- **Android 当前只有单通道**：本结果不能写成 Android 端已具备多通道收益。")
    add("- **不是长期归档验收**：见上一节的 capture 回放边界。")
    add("- **外部实时 push 状态未验证**：本次是离线回放，不含真实在线推送链路。")
    add(
        f"- **最差场必须显式列出**：`{worst['case_id']}`（{worst['selection_basis']}）——all8 口径下 "
        + "，".join(
            f"{METRIC_LABELS[metric].split('（')[0]} {fmt_pct(block['rate'])}"
            for metric, block in worst["metrics"].items()
        )
        + f"。八场汇总通过，但该场仍是明显弱例，应作为后续难例而不是被聚合平均掩盖。"
    )
    if agg["sessions_below_recall_threshold"]:
        listing = "、".join(
            f"`{item['case_id']}` {fmt_pct(item['overlap_recall'])}"
            for item in agg["sessions_below_recall_threshold"]
        )
        add(
            f"- **有 {len(agg['sessions_below_recall_threshold'])}/{coverage['n_selected']} 场的单场 overlap recall 低于 "
            f"{agg['thresholds']['overlap_recall_min']:.2f}**：{listing}。"
            f"按管线定义该指标只在 pooled 层判定（pooled {fmt_pct(gate['overlap_recall']['rate'])} 通过），"
            "但这说明重叠语音检测在部分会议上仍明显不足，是后续必须攻的方向，不能因为聚合通过而不提。"
        )
    add("")

    # ---- 5. 附录 ----
    add("## 5. 附录")
    add("")
    add("### 5.1 研发里程碑（三层证据，不可跨层计算提升）")
    add("")
    add("| 里程碑 | 证据层 | 关键数值 | 评分边界 | 不能证明什么 |")
    add("|---|---|---|---|---|")
    for row in data["milestones"]:
        if row["metrics"] is None:
            value = row["source"].get("note") or "无机器可读源"
        elif row["id"] == "auto_mask_component":
            metrics = row["metrics"]
            value = (
                f"overlap CER {fmt_pct(metrics['overlap_cer_baseline'])} → "
                f"{fmt_pct(metrics['overlap_cer_automatic'])}；"
                f"non-overlap CER {fmt_pct(metrics['non_overlap_cer_automatic'])}；"
                f"前端 RTF {metrics['frontend_rtf']:.4f}；{metrics['cases_improved_vs_ch0']}/8 场改善"
            )
        else:
            metrics = row["metrics"]
            value = "；".join(
                f"{METRIC_LABELS[name].split('（')[0]} {fmt_pct(m['ch0_rate'])} → {fmt_pct(m['all8_rate'])}"
                for name, m in metrics.items()
            )
        add(
            f"| {row['label']} | `{row['tier']}` | {value} | {row['evidence_boundary']} | "
            + "；".join(row["cannot_prove"])
            + " |"
        )
    add("")
    add("三层里程碑的掩码来源、片段边界、连续性和计分范围都不同，"
        "**禁止跨行相减、排名或画趋势图**；只有第 2 节的主表可以做百分点与相对错误降低的计算。")
    add("")

    add("### 5.2 逐场明细（按 all8 cpCER 从差到好）")
    add("")
    add("| 排名 | 会议 | all8 cpCER | ch0 cpCER | all8 overlap | ch0 overlap | all8 non-overlap | ch0 non-overlap |")
    add("|---|---|---|---|---|---|---|---|")
    per_case = data["_per_case"]
    for entry in worst["rankings"]["cpcer"]:
        case_id = entry["case_id"]
        block = per_case[case_id]
        add(
            f"| {entry['rank']} | `{case_id}` | {fmt_pct(block['all8']['cpcer']['rate'])} | "
            f"{fmt_pct(block['ch0']['cpcer']['rate'])} | {fmt_pct(block['all8']['overlap']['rate'])} | "
            f"{fmt_pct(block['ch0']['overlap']['rate'])} | {fmt_pct(block['all8']['non_overlap']['rate'])} | "
            f"{fmt_pct(block['ch0']['non_overlap']['rate'])} |"
        )
    add("")

    add("### 5.3 汇总门禁")
    add("")
    for name, value in sorted(data["summary_gates"].items()):
        add(f"- `{name}`: {value}")
    add("")

    add("### 5.4 输入与 hash（用于复核）")
    add("")
    add("| 角色 | 路径 | SHA-256 | schema |")
    add("|---|---|---|---|")
    for item in data["inputs"]:
        add(f"| {item['role']} | `{item['path']}` | `{item['sha256']}` | `{item['schema']}` |")
    add("")
    add(f"- ASR 模型：{provenance['asr_model_sha256']}")
    add(f"- Sortformer 模型：{provenance['sortformer_model_sha256']}")
    add(f"- 分音变体 / 活动阈值：{provenance['diar_variant']} @ {provenance['activity_threshold']}")
    add("")
    add("### 5.5 复算校验")
    add("")
    add("`full8-summary.json` 发布的每个聚合比率都由 8 场原始分子/分母重新求和复算，"
        "分子、分母和比率三者全部一致（详见 `comparison.json.recompute_cross_check`）。")
    add("")
    return "\n".join(lines) + "\n"


def write_outputs(data: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "comparison.json"
    csv_path = out_dir / "metrics.csv"
    report_path = out_dir / "REPORT.md"

    payload = {key: value for key, value in data.items() if key != "_per_case"}
    comparison_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(data, csv_path)
    report_path.write_text(render_report(data), encoding="utf-8")
    return [comparison_path, csv_path, report_path]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="从已有 JSON 构建 AliMeeting 音频前端优化对比报告数据包（只读，不跑音频/模型）。",
    )
    parser.add_argument("--full8-summary", required=True, type=Path, help="full8 批次的 full8-summary.json")
    parser.add_argument("--batch-manifest", required=True, type=Path, help="full8 批次的 batch-run-manifest.json")
    parser.add_argument("--batch-dir", required=True, type=Path, help="full8 批次根目录（逐场 attempt 目录的父目录）")
    parser.add_argument("--auto-mask-retention", required=True, type=Path, help="auto-mask 组件的 retention.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="输出目录（必须已被 gitignore）")
    parser.add_argument("--sortformer-scores", type=Path, default=None, help="可选：Sortformer 连续分音 scores.json")
    parser.add_argument("--oracle-json", type=Path, default=None, help="可选：oracle 里程碑的机器可读 JSON")
    parser.add_argument("--oracle-note", default=DEFAULT_ORACLE_NOTE, help="oracle 里程碑无机器可读源时的说明文字")
    args = parser.parse_args(argv)

    return Config(
        full8_summary=args.full8_summary,
        batch_manifest=args.batch_manifest,
        batch_dir=args.batch_dir,
        auto_mask_retention=args.auto_mask_retention,
        out_dir=args.out_dir,
        sortformer_scores=args.sortformer_scores,
        oracle_json=args.oracle_json,
        oracle_note=args.oracle_note,
    )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        data = build_comparison(cfg)
    except FailClosed as exc:
        print("FAIL-CLOSED: 报告数据包未生成", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    written = write_outputs(data, cfg.out_dir)
    print(f"OK: 已生成 {len(written)} 个文件 -> {cfg.out_dir}")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
