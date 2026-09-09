"""Thin batch driver for the eight continuous E2E sessions.

One foreground command runs the whole batch: preflight (git HEAD/status + disk),
eight serial sessions (each = wrapper -> verifier -> classify), an atomic
``batch-run-manifest.json`` ledger, and finally the summarizer. The user never
substitutes CASE per session and no background/timer job is created.

    conda run --no-capture-output -n hermes \
      python evals/audio_frontend/run_continuous_full8.py \
        --batch-root reports/p4_continuous_e2e/20260908-full8-run1 \
        --prep-dir reports/p3_continuous_prep/20260907-full8 \
        --predictions reports/p3_sortformer_continuous/20260907-full8-thr030 \
        --source-run reports/eval_ali/eval-ali-e-candidate-fix2 \
        --frontend-env py311 --scorer-env hermes --wrapper-env hermes

Exit codes: 0 = all eight selected and summary gates passed; non-zero = evidence
gap / summary gate failure / preflight failure.

Failure state machine (single source of truth imported from the summarizer):
* evidence failure  -> attempt not selected, batch ABORTS, never retried
  (network / memory / hash drift / WAV / Timeline / product missing).
* quality failure   -> attempt still SELECTED (evidence valid), batch continues and
  ends red.
* infra transient   -> only a driver-captured launch exception or a pre-defined
  stage timeout is retried (<= attempt2); everything else stops (no retry).

This module never runs ASR by itself: it only shells out to the existing wrapper
and verifier. ``--dry-run`` validates git/manifest/commands and prints what would
run, invoking nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from summarize_continuous_full8 import (
    EVIDENCE_CHECKS,
    QUALITY_CHECKS,
    SCHEMA,
    PLAN_ID,
    classify_attempt,
    classify_missing_product,
)

#: SCHEMA and PLAN_ID are defined ONCE, in ``summarize_continuous_full8`` (the
#: module that validates the batch manifest), and re-exported here so the writer and
#: the validator can never drift apart.
assert SCHEMA == "continuous_full8_batch.v3"
assert PLAN_ID == "2026-09-08-audio-continuous-e2e-full8"
MAX_RETRIES_DEFAULT = 2  # attempt0 + retry1 + retry2
#: Locked source roots: every tracked *.py under these directories is part of the
#: runtime and is HEAD-checked (staged+unstaged, relative to HEAD) before the run,
#: after every session, and at the end. We deliberately NO LONGER hand-enumerate a
#: few imports: instead the whole managed source of the relevant run-code trees is
#: locked, so a change to ANY runtime file (e.g. the MVDR enhancer under
#: ``evals/audio_frontend``, or ``agent_bridge`` under ``ai_glasses_memory_assistant``)
#: refuses the run before start or stops mid-batch. ``scripts`` is the repo-root
#: scripts tree that the scorer pulls from via ``ROOT/scripts``.
LOCKED_SOURCE_ROOTS = ("evals/audio_frontend", "ai_glasses_memory_assistant", "scripts")

#: Process-local cache so the pre-run lock + per-session re-checks + end re-check
#: don't re-run ``git ls-files`` repeatedly within one invocation.
_CACHED_MANAGED_SOURCES: Optional[Tuple[str, ...]] = None


def enumerate_managed_sources(repo_root: Path) -> Tuple[str, ...]:
    """Repo-relative paths of every managed ``*.py`` under ``LOCKED_SOURCE_ROOTS``.

    Single source of truth for the source lock: deterministic and derived from git
    (so it always matches what is actually in the tree), covering all three run-code
    trees. Used for the pre-run lock, the post-session drift re-check, and the
    end-of-batch re-check, and recorded into the manifest for audit.

    BOTH tracked and untracked (but not git-ignored) ``*.py`` files are enumerated.
    Enumerating tracked files alone would silently omit brand-new sources — e.g. the
    six full8 entry scripts themselves before they are first committed — so the very
    code doing the locking could escape the lock. Untracked sources are still hashed
    into the manifest, but ``GitInspector.matches_head`` refuses them (a file that is
    not in HEAD cannot be HEAD-clean), so a real run stops until they are committed.
    """
    global _CACHED_MANAGED_SOURCES
    if _CACHED_MANAGED_SOURCES is not None:
        return _CACHED_MANAGED_SOURCES
    out: set = set()
    for extra in ([], ["--others", "--exclude-standard"]):
        proc = subprocess.run(
            ["git", "ls-files", *extra, "--", *LOCKED_SOURCE_ROOTS],
            cwd=str(repo_root), capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.endswith(".py"):
                out.add(line)
    _CACHED_MANAGED_SOURCES = tuple(sorted(out))
    return _CACHED_MANAGED_SOURCES

#: The exact eight case_ids of the locked full8 plan. Order matters: the batch
#: runs them serially in this order and a prep-manifest that lists a different
#: set (e.g. seven sessions, or a renamed id) is refused at preflight.
LOCKED_CASE_IDS = (
    "R8001_M8004-full", "R8003_M8001-full", "R8007_M8010-full", "R8007_M8011-full",
    "R8008_M8013-full", "R8009_M8018-full", "R8009_M8019-full", "R8009_M8020-full",
)

#: Fixed model hashes the scorer enforces (single source of truth mirrored here,
#: NOT read from the current run's own manifest — a run must never self-certify
#: the models it used). Mirrors score_continuous_e2e.ASR_MODEL_SHA256 and
#: diarize_sortformer.MODEL_SHA256; the full8 driver does not import those
#: modules (importing the scorer drags in torch/NeMo).
LOCKED_ASR_MODEL_SHA256 = "12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d"
LOCKED_SORTFORMER_MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"

#: Binary gigabyte. EVERY disk figure in this module is GiB (1024**3 bytes),
#: never the decimal 1e9 "GB" — keep constants, field names, help text, README and
#: the manifest keys in sync so a GiB is never written as a GB (or vice versa).
GIB = 1024 ** 3

#: WAV output characteristics used to DERIVE (not guess) the disk budget. The
#: frontend writes ONE MONO PCM_16 WAV PER ANONYMISED TRACK — not one multi-channel
#: WAV. Sortformer is a fixed 4-speaker model (``diar_streaming_sortformer_4spk-v2.1``)
#: so at most 4 tracks exist, and both channel variants (``all8`` / ``ch0``) are
#: written. WAV upper bound = seconds x 16000 x 2 bytes x 4 tracks x 2 variants.
WAV_SAMPLE_RATE = 16_000
WAV_BYTES_PER_SAMPLE = 2                      # PCM_16, one mono file per track
WAV_MAX_TRACKS = 4                            # Sortformer 4spk model cap
WAV_NUM_VARIANTS = 2                          # all8 + ch0
#: Fixed per-session allowance for text/JSON/log/REPORT artifacts (scores.json,
#: asr.jsonl, logs, REPORT.md). Deliberately small and explicit, not a magic per-session GiB.
TEXT_LOG_MARGIN_GIB_PER_SESSION = 0.05
#: Hard safety margin kept free on the volume regardless of derived product size.
DISK_SAFETY_MARGIN_GIB = 8.0


def derive_disk_budget(
    total_processed_seconds: float,
    *,
    sample_rate: int = WAV_SAMPLE_RATE,
    bytes_per_sample: int = WAV_BYTES_PER_SAMPLE,
    max_tracks: int = WAV_MAX_TRACKS,
    num_variants: int = WAV_NUM_VARIANTS,
    text_log_margin_gib_per_session: float = TEXT_LOG_MARGIN_GIB_PER_SESSION,
    n_sessions: int = 8,
    retry_factor: int = 1,
    safety_margin_gib: float = DISK_SAFETY_MARGIN_GIB,
) -> dict:
    """Derive the disk budget from the ACTUAL output structure (no magic GiB).

    WAV upper bound = total_processed_seconds x sample_rate x bytes_per_sample x
    max_tracks x num_variants. The frontend writes one MONO PCM_16 file per
    anonymised track rather than a single 8-channel file, and Sortformer caps at
    4 speakers, so the upper bound is 4 tracks x 2 variants of mono 16 kHz 16-bit
    audio. On top of WAV bytes we add a fixed text/log margin per session
    (scores.json, asr.jsonl, logs, REPORT.md), then multiply by the retry factor
    (attempt0..attemptN, i.e. retry retention) and add a hard safety margin.
    Source inputs already exist on disk and are NOT re-counted as new space.

    ALL figures are GiB (1024**3 bytes), never decimal GB.

    Returns a dict with every derived term so the result is fully auditable in the
    manifest. ``total_processed_seconds`` must be the sum of the eight sessions'
    processed durations (read from the prep manifest's ``total_audio_seconds``).
    """
    wav_bytes = (total_processed_seconds * sample_rate * bytes_per_sample
                 * max_tracks * num_variants)
    wav_gib = round(wav_bytes / GIB, 6)
    text_log_gib = round(text_log_margin_gib_per_session * n_sessions, 6)
    # Round per-attempt FIRST, then multiply: the recorded terms must literally add
    # up when an auditor recomputes total = per_attempt x retry + safety_margin.
    per_attempt_total_gib = round(wav_gib + text_log_gib, 6)  # one attempt, all sessions
    total_needed_gib = round(per_attempt_total_gib * retry_factor + safety_margin_gib, 6)
    return {
        "total_processed_seconds": float(total_processed_seconds),
        "sample_rate": sample_rate,
        "bytes_per_sample": bytes_per_sample,
        "max_tracks": max_tracks,
        "num_variants": num_variants,
        "wav_gib": round(wav_gib, 6),
        "text_log_gib": round(text_log_gib, 6),
        "per_attempt_total_gib": round(per_attempt_total_gib, 6),
        "retry_factor": int(retry_factor),
        "n_sessions": int(n_sessions),
        "safety_margin_gib": float(safety_margin_gib),
        # NOTE: wav_gib and text_log_gib already cover ALL sessions, so the retry
        # factor is the only multiplier applied on top (an attempt is one full
        # 8-session pass). Writing "x attempts x sessions" here would double-count.
        "derivation": (
            f"wav_upper = {total_processed_seconds:.3f}s (all {n_sessions} sessions) "
            f"x {sample_rate}Hz x {bytes_per_sample}B x {max_tracks} tracks "
            f"x {num_variants} variants = {wav_gib:.3f} GiB; "
            f"text/log = {text_log_margin_gib_per_session} GiB/session x {n_sessions} "
            f"sessions = {text_log_gib:.3f} GiB; "
            f"per attempt (all sessions) = {per_attempt_total_gib:.3f} GiB "
            f"x {retry_factor} attempts (max_retries {retry_factor - 1} + 1, retry retention) "
            f"+ {safety_margin_gib:.1f} GiB safety margin = {total_needed_gib:.3f} GiB"
        ),
        "total_needed_gib": round(total_needed_gib, 6),
    }


def repo_root_of(path: Path) -> Path:
    return path.resolve().parents[2]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --- git inspection (injectable for tests) -----------------------------------

class GitInspector:
    def __init__(self, runner: Callable[[List[str]], "subprocess.CompletedProcess"] = None,
                 repo_root: Optional[Path] = None):
        self.runner = runner or (lambda cmd: subprocess.run(cmd, cwd=str(repo_root) if repo_root else None,
                                                            capture_output=True, text=True))
        self.repo_root = repo_root

    def _run(self, args: List[str]) -> Tuple[int, str]:
        proc = self.runner(["git"] + args)
        return proc.returncode, (proc.stdout or "")

    def head_sha(self) -> str:
        rc, out = self._run(["rev-parse", "HEAD"])
        return out.strip() if rc == 0 else ""

    def status_porcelain(self) -> str:
        rc, out = self._run(["status", "--porcelain"])
        return out.strip() if rc == 0 else ""

    def is_tracked(self, rel: str) -> bool:
        rc, _ = self._run(["ls-files", "--error-unmatch", rel])
        return rc == 0

    def matches_head(self, rel: str) -> bool:
        """True when ``rel`` is tracked and its working-tree bytes equal HEAD.

        ``git diff --quiet HEAD -- <path>`` is the single source of truth: it
        compares the working tree to HEAD (committed content), so a file that is
        only staged, only modified, or modified+staged still counts as drifted.
        Untracked files always fail (rc != 0).
        """
        if not self.is_tracked(rel):
            return False
        rc, _ = self._run(["diff", "--quiet", "HEAD", "--", rel])
        return rc == 0

    def runtime_sources_locked(self, files: List[str]) -> Tuple[bool, List[str]]:
        bad = []
        for f in files:
            if not self.is_tracked(f) or not self.matches_head(f):
                bad.append(f)
        return (len(bad) == 0), bad


# --- manifest atomic write ----------------------------------------------------

def write_manifest_atomic(batch_root: Path, manifest: dict) -> None:
    """Write the full manifest atomically, preserving all old attempt records.

    Same-dir temp file -> flush+fsync -> os.replace (atomic on POSIX). The caller
    is responsible for mutating ``manifest`` (append-only to attempts); this only
    guarantees the on-disk swap never leaves a half-written file and never drops
    history.
    """
    batch_root.mkdir(parents=True, exist_ok=True)
    tmp = batch_root / f"batch-run-manifest.json.tmp.{os.getpid()}"
    data = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, batch_root / "batch-run-manifest.json")


# --- command builders (pure, unit-tested) ------------------------------------

def build_wrapper_command(
    conda: str,
    wrapper_env: str,
    repo_root: Path,
    *,
    prep_dir: Path,
    predictions: Path,
    source_run: Path,
    case_id: str,
    attempt_dir: Path,
    frontend_env: str,
    scorer_env: str,
    replay_timeout: float,
    gate_scope: str = "full",
    diar_variant: str = "micA_mean",
    activity_threshold: float = 0.30,
    enhance_variant: str = "all8",
    baseline_variant: str = "ch0",
) -> List[str]:
    return [
        conda, "run", "--no-capture-output", "-n", wrapper_env,
        "python", str(repo_root / "evals" / "audio_frontend" / "run_full_loop_timed.py"),
        "--prep-dir", str(prep_dir),
        "--predictions", str(predictions),
        "--case-id", case_id,
        "--diar-variant", diar_variant,
        "--activity-threshold", str(activity_threshold),
        "--enhance-variant", enhance_variant,
        "--baseline-variant", baseline_variant,
        "--source-run", str(source_run),
        "--out", str(attempt_dir),
        "--gate-scope", gate_scope,
        "--replay-service",
        "--replay-timeout", str(replay_timeout),
        "--frontend-env", frontend_env,
        "--scorer-env", scorer_env,
    ]


def build_verifier_command(
    conda: str, wrapper_env: str, repo_root: Path, attempt_dir: Path
) -> List[str]:
    return [
        conda, "run", "--no-capture-output", "-n", wrapper_env,
        "python", str(repo_root / "evals" / "audio_frontend" / "verify_frontend_wavs.py"),
        str(attempt_dir / "frontend"),
    ]


# --- run one wrapper (pure-ish, depends on injected runner) -------------------

def run_subprocess(cmd: List[str], runner, timeout: Optional[float]) -> Tuple[str, int]:
    """Return ("completed", exit_code) or ("launch_exception"|"timeout", -1)."""
    try:
        proc = runner(cmd, timeout=timeout)
        return "completed", int(proc.returncode)
    except subprocess.TimeoutExpired:
        return "timeout", -1
    except (OSError, ValueError) as exc:  # e.g. conda not found -> FileNotFoundError
        print(f"[driver] launch exception: {exc}", file=sys.stderr)
        return "launch_exception", -1


# --- per-session processing --------------------------------------------------

def classify_after_run(
    attempt_dir: Path,
    wrapper_exit: int,
    verifier_exit: Optional[int],
    expected_hashes: Optional[dict],
    case_id: Optional[str] = None,
) -> dict:
    """Re-derive status from real product files (never from exit code / dir name).

    ``verifier_exit`` is fail-closed: ``None`` means the WAV verifier did not produce
    a usable result, so the WAV/segment evidence is *unverified* and the attempt is
    treated as an evidence failure. It is never assumed to have passed.
    """
    scores_path = attempt_dir / "scores.json"
    timing_path = attempt_dir / "full-loop-timing.json"
    if not scores_path.is_file():
        # wrapper returned but no scores.json -> product missing / crashed before write
        return classify_missing_product(f"wrapper_exit={wrapper_exit}, scores.json absent")
    try:
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return classify_missing_product("scores.json unparseable")
    timing = None
    if timing_path.is_file():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            timing = None
    verifier_passed = (verifier_exit == 0)
    return classify_attempt(scores, timing, verifier_passed, expected_hashes,
                            case_id=case_id, timing_path=timing_path)


def read_expected_hashes(source_run: Path, predictions: Path) -> Tuple[Dict[str, str], List[str]]:
    """Read the model hashes the scorer enforces, from plain JSON manifests.

    * ASR:        ``<source-run>/run-manifest.json`` ->
      ``runtime.ambient_audio_profile.asr_model_sha256``
    * Sortformer: ``<predictions>/run-manifest.json`` -> ``model.sha256``

    These are the same authoritative values ``score_continuous_e2e`` checks against its
    own constants. Importing that module to read the constants directly would drag in
    torch / NeMo, and when the import fails the binding silently disappears
    (fail-open). Reading the manifests keeps the check alive in any environment.

    Returns ``(hashes, problems)``; a non-empty ``problems`` list is a preflight failure.
    A manifest may never certify itself: each hash it declares must equal the
    fixed locked constant (LOCKED_ASR_MODEL_SHA256 / LOCKED_SORTFORMER_MODEL_SHA256),
    otherwise the run refuses to start.
    """
    hashes: Dict[str, str] = {}
    problems: List[str] = []

    src_manifest = source_run / "run-manifest.json"
    if not src_manifest.is_file():
        problems.append(f"source-run manifest missing: {src_manifest}")
    else:
        try:
            m = json.loads(src_manifest.read_text(encoding="utf-8"))
            profile = ((m.get("runtime") or {}).get("ambient_audio_profile") or {})
            sha = profile.get("asr_model_sha256")
            if not sha:
                problems.append(f"asr_model_sha256 absent in {src_manifest}")
            elif sha != LOCKED_ASR_MODEL_SHA256:
                problems.append(
                    f"asr_model_sha256 {sha[:12]}… != locked {LOCKED_ASR_MODEL_SHA256[:12]}… "
                    f"(a run cannot self-certify its own models; fix the source run, not the lock)")
            else:
                hashes["asr_model_sha256"] = sha
        except (ValueError, OSError) as exc:
            problems.append(f"source-run manifest unreadable: {exc}")

    pred_manifest = predictions / "run-manifest.json"
    if not pred_manifest.is_file():
        problems.append(f"predictions manifest missing: {pred_manifest}")
    else:
        try:
            m = json.loads(pred_manifest.read_text(encoding="utf-8"))
            sha = (m.get("model") or {}).get("sha256")
            if not sha:
                problems.append(f"model.sha256 absent in {pred_manifest}")
            elif sha != LOCKED_SORTFORMER_MODEL_SHA256:
                problems.append(
                    f"model.sha256 {sha[:12]}… != locked {LOCKED_SORTFORMER_MODEL_SHA256[:12]}… "
                    f"(a run cannot self-certify its own models; fix the predictions, not the lock)")
            else:
                hashes["sortformer_model_sha256"] = sha
        except (ValueError, OSError) as exc:
            problems.append(f"predictions manifest unreadable: {exc}")

    return hashes, problems


def decide_transient(result_kind: str) -> bool:
    """True only for a driver-captured launch exception or a pre-defined timeout."""
    return result_kind in ("launch_exception", "timeout")


def process_case(
    case_id: str,
    *,
    batch_root: Path,
    conda: str,
    wrapper_env: str,
    repo_root: Path,
    prep_dir: Path,
    predictions: Path,
    source_run: Path,
    frontend_env: str,
    scorer_env: str,
    replay_timeout: float,
    expected_hashes: Optional[dict],
    max_retries: int,
    stage_timeout: Optional[float],
    runner,
    dry_run: bool,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[str], List[dict], bool, Optional[str]]:
    """Run one case serially across attempts.

    Returns (selected_attempt_dir, attempt_records, aborted, abort_reason).
    ``aborted`` is True when an evidence gap forces the whole batch to stop.

    ``should_stop`` reports a received SIGINT/SIGTERM. A signal that kills the wrapper
    mid-session would otherwise surface as a bogus ``evidence_fail`` (scores.json never
    written); it is recorded as ``interrupted`` instead, and never retried.
    """
    _stopped = should_stop or (lambda: False)
    attempt_records: List[dict] = []
    selected: Optional[str] = None
    aborted = False
    abort_reason: Optional[str] = None

    for attempt in range(0, max_retries + 1):
        if not dry_run and _stopped():
            abort_reason = f"{case_id}: interrupted before attempt{attempt} (no auto-restart)"
            aborted = True
            break
        # Relative-to-batch-root path is the ONLY form persisted in the manifest:
        # ``<case_id>/attemptN``. The summarizer resolves it against the batch root
        # and refuses anything that escapes it, so a relative path can never be
        # mis-joined into a second root by a later consumer.
        rel_attempt_dir = Path(case_id) / f"attempt{attempt}"
        attempt_dir = batch_root / rel_attempt_dir
        wrapper_cmd = build_wrapper_command(
            conda, wrapper_env, repo_root,
            prep_dir=prep_dir, predictions=predictions, source_run=source_run,
            case_id=case_id, attempt_dir=attempt_dir,
            frontend_env=frontend_env, scorer_env=scorer_env,
            replay_timeout=replay_timeout,
        )
        verifier_cmd = build_verifier_command(conda, wrapper_env, repo_root, attempt_dir)

        if dry_run:
            print(f"[dry-run][{case_id}] attempt{attempt} would run:")
            print("  " + " ".join(wrapper_cmd))
            print("  " + " ".join(verifier_cmd))
            # No ASR/verifier/wrapper invoked. Record a planned placeholder.
            attempt_records.append({
                "attempt": attempt, "attempt_dir": str(rel_attempt_dir),
                "exit_code": None, "verifier_exit": None, "verifier_passed": None,
                "status": {"evidence_valid": None, "quality_passed": None,
                           "selected": None, "failure_class": "planned", "failure_reasons": []},
            })
            break  # dry-run only previews the first attempt

        # 1) wrapper
        kind, code = run_subprocess(wrapper_cmd, runner, stage_timeout)
        if _stopped():
            # A signal reached the process group and killed the wrapper. Record the
            # real cause instead of the evidence_fail it would otherwise look like.
            attempt_records.append({
                "attempt": attempt, "attempt_dir": str(rel_attempt_dir),
                "exit_code": code if kind == "completed" else None,
                "verifier_exit": None, "verifier_passed": None,
                "status": {"evidence_valid": False, "quality_passed": False,
                           "selected": False, "failure_class": "interrupted",
                           "failure_reasons": ["interrupted:signal_received"]},
            })
            abort_reason = f"{case_id}: interrupted during attempt{attempt} (no auto-restart)"
            aborted = True
            break
        if kind == "completed" and code == 5:
            # output dir already exists (shouldn't happen with fresh dirs) -> safety stop
            abort_reason = f"{case_id}: wrapper refused existing --out (exit 5)"
            aborted = True
            break

        # 2) verifier — run whenever the frontend produced a directory, REGARDLESS of
        #    the wrapper exit code. A quality failure makes the wrapper exit 2 while
        #    the WAVs and scores.json are perfectly valid products that still need
        #    their WAV/segment evidence verified before the attempt can be selected.
        verifier_exit: Optional[int] = None
        if kind == "completed" and attempt_dir.joinpath("frontend").is_dir():
            vk, ve = run_subprocess(verifier_cmd, runner, None)
            verifier_exit = ve if vk == "completed" else None

        # 3) classify from products
        status = classify_after_run(attempt_dir, code if kind == "completed" else -1,
                                    verifier_exit, expected_hashes, case_id=case_id)
        attempt_records.append({
            "attempt": attempt, "attempt_dir": str(rel_attempt_dir),
            "exit_code": code if kind == "completed" else None,
            "verifier_exit": verifier_exit,
            "verifier_passed": (verifier_exit == 0) if verifier_exit is not None else None,
            "status": status,
        })

        # 4) selection / retry / abort decisions
        if status["evidence_valid"]:
            selected = str(rel_attempt_dir)
            break  # quality_fail still selected -> done with this case
        if kind in ("launch_exception", "timeout"):
            if attempt < max_retries:
                print(f"[driver][{case_id}] transient {kind}; retrying attempt{attempt+1}", file=sys.stderr)
                continue
            abort_reason = f"{case_id}: transient {kind} exhausted retries"
            aborted = True
            break
        # evidence_fail or product_missing (infra_fail): never auto-retry per §13
        abort_reason = f"{case_id}: {status['failure_class']} ({'; '.join(status['failure_reasons'][:3])})"
        aborted = True
        break

    return selected, attempt_records, aborted, abort_reason


# --- preflight ---------------------------------------------------------------

def nearest_existing_ancestor(path: Path) -> Path:
    """Walk up from ``path`` to the nearest existing ancestor (for df / disk)."""
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def disk_free_gib(path: Path) -> Optional[float]:
    """Free space on the filesystem holding ``path``, in GiB; None if unavailable."""
    try:
        import shutil
        return shutil.disk_usage(nearest_existing_ancestor(path)).free / (1024 ** 3)
    except (OSError, ValueError):
        return None


def preflight(
    repo_root: Path,
    batch_root: Path,
    git: GitInspector,
    enforce_git: bool,
    *,
    expected_hashes: Optional[dict] = None,
    locked_case_ids: Tuple[str, ...] = LOCKED_CASE_IDS,
    free_disk_gib: Optional[float] = None,
    total_processed_seconds: Optional[float] = None,
    max_retries: int = MAX_RETRIES_DEFAULT,
) -> dict:
    head = git.head_sha()
    status = git.status_porcelain()
    sources = [str(repo_root / f) for f in enumerate_managed_sources(repo_root)]
    locked, bad = git.runtime_sources_locked(sources)
    source_sha: Dict[str, str] = {}
    for f in enumerate_managed_sources(repo_root):
        p = repo_root / f
        if p.is_file():
            source_sha[f] = sha256_of(p)
    # Model hashes must be the two fixed locks; a manifest may never self-certify.
    hash_problems: List[str] = []
    if expected_hashes:
        if expected_hashes.get("asr_model_sha256") != LOCKED_ASR_MODEL_SHA256:
            hash_problems.append("asr model hash != locked constant")
        if expected_hashes.get("sortformer_model_sha256") != LOCKED_SORTFORMER_MODEL_SHA256:
            hash_problems.append("sortformer model hash != locked constant")
    disk = ""
    try:
        proc = subprocess.run(["df", "-h", str(nearest_existing_ancestor(batch_root))],
                              capture_output=True, text=True)
        disk = proc.stdout.strip() or "df returned no output"
    except (OSError, ValueError):
        disk = "df unavailable"
    avail_gib = free_disk_gib if free_disk_gib is not None else disk_free_gib(batch_root)

    # --- disk budget derivation ------------------------------------------------
    # Derived from the ACTUAL output structure (never a magic per-session GiB):
    # WAV upper bound from the eight sessions' processed duration, output sample
    # rate, bytes/sample, the Sortformer 4-track cap and the all8/ch0 two variants
    # (one MONO PCM_16 file per track), plus a fixed text/log margin, scaled by the
    # retry factor, with a hard safety margin. Every term is recorded in the
    # manifest. Source inputs already exist on disk and are NOT re-counted.
    # Existing products are NEVER deleted to satisfy the budget.
    retry_factor = max(0, int(max_retries)) + 1
    n_sessions = len(locked_case_ids)
    budget_params_ok = (total_processed_seconds is not None
                        and total_processed_seconds > 0
                        and isinstance(total_processed_seconds, (int, float))
                        and not isinstance(total_processed_seconds, bool))
    if budget_params_ok:
        disk_budget_basis = derive_disk_budget(
            float(total_processed_seconds),
            n_sessions=n_sessions, retry_factor=retry_factor,
            safety_margin_gib=DISK_SAFETY_MARGIN_GIB)
        total_needed_gib = disk_budget_basis["total_needed_gib"]
    else:
        # Illegal/unknown processed-duration parameter: cannot derive a budget.
        total_needed_gib = float("inf")
        disk_budget_basis = {
            "total_processed_seconds": total_processed_seconds,
            "n_sessions": n_sessions, "retry_factor": retry_factor,
            "derivation": "INVALID: total_processed_seconds missing or <= 0",
            "total_needed_gib": total_needed_gib,
        }
    # Unreadable available space (None), illegal budget params, or insufficient
    # space MUST NOT be marked budget_ok.
    disk_budget_ok = bool(
        budget_params_ok and avail_gib is not None and avail_gib >= total_needed_gib)
    disk_budget_basis["available_gib"] = avail_gib
    disk_budget_basis["budget_ok"] = disk_budget_ok
    disk_budget_basis["params_ok"] = budget_params_ok
    if enforce_git and not budget_params_ok:
        raise SystemExit(
            f"disk budget cannot be derived: total_processed_seconds="
            f"{total_processed_seconds!r} (must be > 0; read from prep "
            f"total_audio_seconds or window_*_s)")
    if enforce_git and avail_gib is None:
        # Available space unreadable -> must refuse the real run, never mark ok.
        raise SystemExit(
            "disk budget cannot be checked: available free space unreadable")
    if enforce_git and avail_gib is not None and not disk_budget_ok:
        # A space-starved run fails fast with the clearest cause. Checked before the
        # git lock so it aborts before any byte is written.
        raise SystemExit(
            f"insufficient disk space: {avail_gib:.3f} GiB free, need "
            f">= {total_needed_gib:.3f} GiB ({disk_budget_basis['derivation']})")
    # A real run requires the WHOLE working tree to be clean, not only the managed
    # Python sources: a modified or untracked non-Python file (config, data, notes)
    # would otherwise be discovered only after eight sessions have been produced.
    if enforce_git and (status or "").strip():
        raise SystemExit(
            "git working tree not clean; commit or stash everything before a real run: "
            + " | ".join((status or "").strip().splitlines()[:10])
            + (" ..." if len((status or "").strip().splitlines()) > 10 else ""))
    if enforce_git and not locked:
        raise SystemExit(f"git lock failed; runtime sources not tracked/clean at HEAD: {bad}")
    if enforce_git and hash_problems:
        raise SystemExit("model-hash lock failed: " + "; ".join(hash_problems))
    return {
        "head_sha": head, "status": status, "runtime_sources_locked": locked,
        "runtime_source_sha256": source_sha, "disk_df_h": disk,
        "disk_free_gib": avail_gib,
        "disk_budget_ok": disk_budget_ok,
        "disk_budget_basis": disk_budget_basis,
        "model_hashes_locked": not hash_problems,
    }


# --- manifest init -----------------------------------------------------------

def init_manifest(args: argparse.Namespace, prep: dict, prefs: dict) -> dict:
    cases = []
    per_case_hashes: Dict[str, dict] = {}
    for c in prep.get("cases", []):
        cid = c.get("case_id")
        if not cid:
            continue
        cases.append({"order": len(cases), "case_id": cid})
        per_case_hashes[cid] = {
            "cut_wav_sha256": c.get("cut_wav_sha256"),
            "textgrid_sha256": c.get("textgrid_sha256"),
        }
    manifest = {
        "schema": SCHEMA,
        "plan_id": PLAN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"head_sha": prefs.get("head_sha"), "status": prefs.get("status"),
                "runtime_sources_match_head": prefs.get("runtime_sources_locked"),
                "runtime_source_sha256": prefs.get("runtime_source_sha256")},
        "locks": {
            "prep_dir": str(args.prep_dir),
            "predictions": str(args.predictions),
            "source_run": str(args.source_run),
            "diar_variant": "micA_mean", "activity_threshold": 0.30,
            "enhance_variant": "all8", "baseline_variant": "ch0",
            "gate_scope": "full", "replay_timeout": args.replay_timeout,
            "model_hashes": {
                "asr_model_sha256": prefs.get("asr_model_sha256"),
                "sortformer_model_sha256": prefs.get("sortformer_model_sha256"),
            },
            "prep_manifest_sha256": prefs.get("prep_manifest_sha256"),
            "prediction_run_manifest_sha256": prefs.get("prediction_run_manifest_sha256"),
            "source_run_manifest_sha256": prefs.get("source_run_manifest_sha256"),
            "per_case_hashes": per_case_hashes,
        },
        "env": {
            "frontend_env": args.frontend_env, "scorer_env": args.scorer_env,
            "wrapper_env": args.wrapper_env, "conda": prefs.get("conda"),
            "frontend_python": prefs.get("frontend_python"),
            "frontend_python_version": prefs.get("frontend_python_version"),
            "scorer_python": prefs.get("scorer_python"),
            "scorer_python_version": prefs.get("scorer_python_version"),
            "wrapper_python": prefs.get("wrapper_python"),
            "wrapper_python_version": prefs.get("wrapper_python_version"),
        },
        # Disk budget derived from the real output structure: every term of the
        # derivation, the available space and the computed budget are persisted so the
        # run is auditable. Nothing is ever deleted to satisfy it.
        "disk_budget": {
            "total_processed_seconds": (prefs.get("disk_budget_basis") or {}).get("total_processed_seconds"),
            "sample_rate": (prefs.get("disk_budget_basis") or {}).get("sample_rate"),
            "bytes_per_sample": (prefs.get("disk_budget_basis") or {}).get("bytes_per_sample"),
            "max_tracks": (prefs.get("disk_budget_basis") or {}).get("max_tracks"),
            "num_variants": (prefs.get("disk_budget_basis") or {}).get("num_variants"),
            "wav_gib": (prefs.get("disk_budget_basis") or {}).get("wav_gib"),
            "text_log_gib": (prefs.get("disk_budget_basis") or {}).get("text_log_gib"),
            "per_attempt_total_gib": (prefs.get("disk_budget_basis") or {}).get("per_attempt_total_gib"),
            "retry_factor": (prefs.get("disk_budget_basis") or {}).get("retry_factor"),
            "n_sessions": (prefs.get("disk_budget_basis") or {}).get("n_sessions"),
            "safety_margin_gib": (prefs.get("disk_budget_basis") or {}).get("safety_margin_gib"),
            "derivation": (prefs.get("disk_budget_basis") or {}).get("derivation"),
            "available_gib": prefs.get("disk_free_gib"),
            "needed_gib": (prefs.get("disk_budget_basis") or {}).get("total_needed_gib"),
            "params_ok": (prefs.get("disk_budget_basis") or {}).get("params_ok"),
            "budget_ok": prefs.get("disk_budget_ok"),
        },
        "cases": cases,
        "attempts": {},
        "post_run": {},
    }
    return manifest


# --- main --------------------------------------------------------------------

def _find_conda(explicit: Optional[str]) -> str:
    """Locate the conda executable without hardcoding any personal path.

    Order: explicit ``--conda`` -> ``$CONDA_EXE`` -> ``conda`` on PATH ->
    ``$CONDA_PREFIX``-relative (works inside an activated env, including a
    non-base one). Anything else is an explicit error, not a guess.
    """
    from shutil import which
    if explicit:
        return explicit
    candidates: List[Optional[str]] = [os.environ.get("CONDA_EXE"), which("conda")]
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        candidates += [str(p / "bin" / "conda"), str(p.parent.parent / "bin" / "conda")]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    raise SystemExit("conda not found; pass --conda /path/to/conda")


# --- environment probing (injectable for tests) ------------------------------

class EnvProbe:
    """Resolve conda + the python interpreter of each used env.

    ``probe(conda, env)`` runs ``conda run -n <env> python -c ...`` inside the env
    and returns ``(python_path, version)`` — the manifest records these so the
    environment is never a null/unknown field. Injectable so unit tests never
    invoke conda.
    """

    def __init__(self, runner: Optional[Callable[[List[str]], "subprocess.CompletedProcess"]] = None):
        self.runner = runner

    def __call__(self, conda: str, env: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            if self.runner is not None:
                proc = self.runner([conda, "run", "--no-capture-output", "-n", env,
                                    "python", "-c", "import sys;print(sys.executable);print(sys.version.split()[0])"])
            else:
                proc = subprocess.run(
                    [conda, "run", "--no-capture-output", "-n", env,
                     "python", "-c", "import sys;print(sys.executable);print(sys.version.split()[0])"],
                    capture_output=True, text=True, timeout=120)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None, None
        if proc.returncode != 0:
            return None, None
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, None
        return lines[0], lines[1]


def run_env_snapshot(git: GitInspector, repo_root: Path,
                     sources: Optional[Tuple[str, ...]] = None) -> dict:
    """Re-check HEAD / working tree / runtime source hashes (mid-batch drift).

    Uses the same directory-enumerated source set as the pre-run lock when
    ``sources`` is not supplied, so the three checks (before / after each session
    / end) are guaranteed identical.
    """
    if sources is None:
        sources = enumerate_managed_sources(repo_root)
    return {
        "head_sha": git.head_sha(),
        "status": git.status_porcelain(),
        "runtime_sources_locked": git.runtime_sources_locked(
            [str(repo_root / f) for f in sources])[0],
        "runtime_source_sha256": {f: sha256_of(repo_root / f) for f in sources
                                  if (repo_root / f).is_file()},
    }


def hashes_drifted(before: dict, after: dict) -> List[str]:
    """Compare two run_env_snapshot dicts; return a human list of what moved."""
    drift: List[str] = []
    if after.get("head_sha") != before.get("head_sha"):
        drift.append(f"HEAD moved {str(before.get('head_sha'))[:12]} -> {str(after.get('head_sha'))[:12]}")
    if after.get("runtime_sources_locked") is not True:
        drift.append("runtime sources no longer tracked/clean at HEAD")
    b = before.get("runtime_source_sha256") or {}
    a = after.get("runtime_source_sha256") or {}
    for rel in sorted(set(b) | set(a)):
        if a.get(rel) != b.get(rel):
            drift.append(f"runtime source changed: {rel}")
    return drift


def _kill_process_group(pgid: int, signum: int) -> None:
    """Best-effort kill of the whole process group ``pgid``.

    The wrapper (conda run -> frontend/scorer) is started with
    ``start_new_session=True`` so it owns the group; killing the GROUP — not just
    the direct child — is what guarantees no orphaned decoder/scorer survives a
    timeout or interrupt.
    """
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def build_forwarding_runner(stop_flag: dict, child: dict):
    """A subprocess.run-compatible runner that forwards SIGINT/SIGTERM.

    The wrapper (conda run -> frontend/scorer) is started with
    ``start_new_session=True`` so it gets its own process group; on a signal the
    driver kills that whole group and marks ``stop_flag``. The child's exit is
    then awaited before the batch records ``interrupted`` — no new session starts,
    nothing is retried.
    """

    def _forward(signum, frame):
        stop_flag["flag"] = True
        proc = child.get("proc")
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc.pid, signum)

    def _run(cmd, timeout=None):
        proc = subprocess.Popen(cmd, start_new_session=True)
        child["proc"] = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the ENTIRE process group (conda run -> python -> decoder), not
            # just the direct child, then reap it before returning. The caller
            # may retry only once the group is confirmed gone (no orphan left to
            # compete for the GPU / port on the next attempt).
            _kill_process_group(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    return _forward, _run


def main(argv: Optional[List[str]] = None, *, runner=None,
         git: Optional[GitInspector] = None, clock=time.time,
         env_probe: Optional[EnvProbe] = None,
         free_disk_gib: Optional[float] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--prep-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--frontend-env", default="py311")
    parser.add_argument("--scorer-env", default="hermes")
    parser.add_argument("--wrapper-env", default="hermes")
    parser.add_argument("--replay-timeout", type=float, default=60.0)
    parser.add_argument("--conda", default=None)
    parser.add_argument("--repo-root", default=None, type=Path)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--stage-timeout", type=float, default=None,
                        help="Optional per-wrapper wall-clock timeout; expiry counts as a transient retry.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate git/manifest/commands and print them; invoke nothing.")
    args = parser.parse_args(argv)
    # max-retries is hard-limited to 0..2 (group 4): attempt0 + up to 2 retries;
    # a higher value is silently clamped, a negative one to 0 (no retry at all).
    args.max_retries = max(0, min(2, int(args.max_retries)))

    repo_root = args.repo_root.resolve() if args.repo_root else repo_root_of(Path(__file__))
    conda = _find_conda(args.conda)
    git_insp = git or GitInspector(repo_root=repo_root)
    dry_run = args.dry_run

    # read prep-manifest for case order + per-case hashes
    prep_manifest_path = args.prep_dir / "prep-manifest.json"
    if not prep_manifest_path.is_file():
        raise SystemExit(f"prep-manifest not found: {prep_manifest_path}")
    prep = json.loads(prep_manifest_path.read_text(encoding="utf-8"))

    # exactly the locked eight sessions must be scheduled, in the locked order.
    # Running seven silently, renaming a case or reordering is an evidence gap:
    # there is no entry point that relaxes this to seven.
    case_ids = [c.get("case_id") for c in (prep.get("cases") or []) if c.get("case_id")]
    if tuple(case_ids) != LOCKED_CASE_IDS:
        raise SystemExit(
            f"prep-manifest cases != the locked eight: got {case_ids}, "
            f"expected {list(LOCKED_CASE_IDS)} (exact set and order; no 7-case runs)"
        )

    # Derive the total processed duration for the disk budget from the prep manifest
    # (it already records total_audio_seconds; fall back to per-case window delta).
    # Source inputs already exist on disk, so they are NOT re-counted as new space.
    total_processed_seconds = prep.get("total_audio_seconds")
    if not total_processed_seconds or total_processed_seconds <= 0:
        cases = prep.get("cases") or []
        try:
            total_processed_seconds = sum(
                (c.get("window_end_s") or 0) - (c.get("window_start_s") or 0)
                for c in cases)
        except Exception:
            total_processed_seconds = None
    if not total_processed_seconds or total_processed_seconds <= 0:
        raise SystemExit(
            "cannot derive disk budget: prep manifest has no total_audio_seconds / "
            "window_*_s (the eight sessions' processed duration is required)")

    # Expected model hashes read from the authoritative manifests (no heavy import,
    # never fail-open). A missing or non-locked hash is a preflight failure.
    expected_hashes, hash_problems = read_expected_hashes(args.source_run, args.predictions)
    if hash_problems:
        raise SystemExit("model-hash binding failed: " + "; ".join(hash_problems))

    probe = env_probe or EnvProbe()
    frontend_python, frontend_pyver = probe(conda, args.frontend_env)
    scorer_python, scorer_pyver = probe(conda, args.scorer_env)
    wrapper_python, wrapper_pyver = probe(conda, args.wrapper_env)
    # Environment probe failure must stop BEFORE any wrapper is launched (group 6):
    # a null interpreter means the env is unusable, so the batch refuses to start
    # rather than running against an undefined python.
    if frontend_python is None or scorer_python is None or wrapper_python is None:
        raise SystemExit(
            "environment probe failed: could not resolve a python interpreter for "
            f"frontend_env={args.frontend_env} scorer_env={args.scorer_env} "
            f"wrapper_env={args.wrapper_env}; refusing to launch the wrapper")
    prefs = {
        "head_sha": git_insp.head_sha(),
        "status": git_insp.status_porcelain(),
        "runtime_sources_locked": git_insp.runtime_sources_locked(
            [str(repo_root / f) for f in enumerate_managed_sources(repo_root)])[0],
        "runtime_source_sha256": {f: sha256_of(repo_root / f) for f in enumerate_managed_sources(repo_root) if (repo_root / f).is_file()},
        "asr_model_sha256": expected_hashes.get("asr_model_sha256"),
        "sortformer_model_sha256": expected_hashes.get("sortformer_model_sha256"),
        "prep_manifest_sha256": sha256_of(prep_manifest_path) if prep_manifest_path.is_file() else None,
        "conda": conda,
        "frontend_python": frontend_python,
        "frontend_python_version": frontend_pyver,
        "scorer_python": scorer_python,
        "scorer_python_version": scorer_pyver,
        "wrapper_python": wrapper_python,
        "wrapper_python_version": wrapper_pyver,
    }
    pred_rm = args.predictions / "run-manifest.json"
    prefs["prediction_run_manifest_sha256"] = sha256_of(pred_rm) if pred_rm.is_file() else None
    src_rm = args.source_run / "run-manifest.json"
    prefs["source_run_manifest_sha256"] = sha256_of(src_rm) if src_rm.is_file() else None

    if not dry_run:
        # Refuse an already-existing batch root BEFORE any byte is written: the
        # driver must never append to or resume a previous batch's manifest.
        if args.batch_root.exists():
            raise SystemExit(
                f"batch root already exists, refusing to write into it: {args.batch_root}\n"
                "choose a NEW --batch-root directory for this batch."
            )
        pf = preflight(repo_root, args.batch_root, git_insp, enforce_git=True,
                  expected_hashes=expected_hashes, free_disk_gib=free_disk_gib,
                  total_processed_seconds=total_processed_seconds,
                  max_retries=args.max_retries)
    else:
        pf = preflight(repo_root, args.batch_root, git_insp, enforce_git=False,
                       expected_hashes=expected_hashes, free_disk_gib=free_disk_gib,
                       total_processed_seconds=total_processed_seconds,
                       max_retries=args.max_retries)
        print(f"[dry-run] git HEAD={pf['head_sha'][:12]} locked={pf['runtime_sources_locked']}")
        print(f"[dry-run] disk: free={pf['disk_free_gib']} budget_ok={pf['disk_budget_ok']} params_ok={pf['disk_budget_basis'].get('params_ok')}")
        print(f"[dry-run] disk budget: {pf['disk_budget_basis']['derivation']} -> {pf['disk_budget_basis']['total_needed_gib']} GiB")

    # Persist the derived disk budget, availability and the REAL git status into the
    # manifest so the run is auditable end-to-end (groups 3 & 5).
    prefs["disk_free_gib"] = pf["disk_free_gib"]
    prefs["disk_budget_basis"] = pf["disk_budget_basis"]
    prefs["disk_budget_ok"] = pf["disk_budget_ok"]
    prefs["git_status"] = pf["status"]

    manifest = init_manifest(args, prep, prefs)
    if dry_run:
        print("[dry-run] batch-run-manifest.json PREVIEW (not written):")
        print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])
        # Print the exact command pair for every case. ``process_case(dry_run=True)``
        # previews attempt0 only and calls the runner zero times.
        def _never(cmd, timeout=None):  # pragma: no cover - guarded by unit test
            raise AssertionError("dry-run must not invoke anything")
        for case in manifest["cases"]:
            process_case(
                case["case_id"], batch_root=args.batch_root, conda=conda,
                wrapper_env=args.wrapper_env, repo_root=repo_root, prep_dir=args.prep_dir,
                predictions=args.predictions, source_run=args.source_run,
                frontend_env=args.frontend_env, scorer_env=args.scorer_env,
                replay_timeout=args.replay_timeout, expected_hashes=expected_hashes,
                max_retries=args.max_retries, stage_timeout=args.stage_timeout,
                runner=_never, dry_run=True,
            )
        print(f"[dry-run] {len(manifest['cases'])} cases planned; "
              f"git locked={prefs['runtime_sources_locked']}")
        if not prefs["runtime_sources_locked"]:
            print("[dry-run] WARNING: runtime sources are not tracked/clean at HEAD; "
                  "the real run will refuse to start until they are committed.")
        print("[dry-run] NO wrapper/verifier/ASR invoked.")
        return 0

    write_manifest_atomic(args.batch_root, manifest)

    # graceful interrupt: forward the signal to the running wrapper's process
    # group, then stop. No auto-restart, no next session.
    stop_flag = {"flag": False}
    child = {"proc": None}
    if runner is None:
        _handler, real_runner = build_forwarding_runner(stop_flag, child)
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        run_callable = real_runner
    else:
        # Test-injected runner: keep the interrupt marker contract (a runner may
        # simulate the signal by raising after setting stop_flag["flag"]).
        run_callable = runner
        old_int = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__("flag", True))
        old_term = signal.signal(signal.SIGTERM, lambda s, f: stop_flag.__setitem__("flag", True))

    batch_aborted = False
    abort_msg = ""
    snapshot_before = run_env_snapshot(git_insp, repo_root)
    drift_messages: List[str] = []
    try:
        for case in manifest["cases"]:
            cid = case["case_id"]
            if stop_flag["flag"]:
                abort_msg = "interrupted by signal"
                batch_aborted = True
                break
            sel, recs, aborted, reason = process_case(
                cid, batch_root=args.batch_root, conda=conda, wrapper_env=args.wrapper_env,
                repo_root=repo_root, prep_dir=args.prep_dir, predictions=args.predictions,
                source_run=args.source_run, frontend_env=args.frontend_env,
                scorer_env=args.scorer_env, replay_timeout=args.replay_timeout,
                expected_hashes=expected_hashes, max_retries=args.max_retries,
                stage_timeout=args.stage_timeout, runner=run_callable, dry_run=False,
                should_stop=lambda: stop_flag["flag"],
            )
            manifest["attempts"].setdefault(cid, {})["attempts"] = recs
            manifest["attempts"][cid]["selected_attempt_dir"] = sel
            write_manifest_atomic(args.batch_root, manifest)
            # Post-session drift re-check: a HEAD move / dirty tree / changed
            # runtime source mid-batch stops the batch; no green summary may be
            # produced from drifting code.
            drift = hashes_drifted(snapshot_before, run_env_snapshot(git_insp, repo_root))
            if drift:
                drift_messages.extend(drift)
                abort_msg = "code drift after " + cid + ": " + "; ".join(drift)
                batch_aborted = True
                break
            if aborted:
                batch_aborted = True
                abort_msg = reason or "evidence/abort"
                break
    except KeyboardInterrupt:
        # A signal the forwarding runner could not swallow (e.g. the wrapper was
        # between launches): the child is gone and the batch stops, recorded as
        # interrupted — never evidence_fail, never auto-restarted.
        batch_aborted = True
        abort_msg = "interrupted by signal (KeyboardInterrupt)"
    finally:
        if runner is None:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        else:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

    # end-of-batch drift re-check + post_run record
    end_snapshot = run_env_snapshot(git_insp, repo_root)
    end_drift = hashes_drifted(snapshot_before, end_snapshot)
    if end_drift:
        drift_messages.extend(end_drift)
        batch_aborted = True
        if not abort_msg:
            abort_msg = "code drift at batch end: " + "; ".join(end_drift)
    manifest["post_run"] = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head_sha": end_snapshot["head_sha"],
        "status": "aborted" if batch_aborted else "completed",
        "code_drift": drift_messages,
        "git_status": end_snapshot["status"],
        "runtime_sources_locked": end_snapshot["runtime_sources_locked"],
        "runtime_source_sha256": end_snapshot["runtime_source_sha256"],
    }
    write_manifest_atomic(args.batch_root, manifest)

    if batch_aborted:
        print(f"[driver] BATCH ABORTED: {abort_msg}", file=sys.stderr)
        return 2

    # every case selected?
    missing = [c["case_id"] for c in manifest["cases"]
               if not manifest["attempts"].get(c["case_id"], {}).get("selected_attempt_dir")]
    if missing:
        print(f"[driver] evidence gap: no selected attempt for {missing}", file=sys.stderr)
        return 2

    # run summarizer on the selected attempts
    from summarize_continuous_full8 import main as summarize_main
    summary_path = args.batch_root / "full8-summary.json"
    rc = summarize_main(["--manifest", str(args.batch_root / "batch-run-manifest.json"),
                         "--out", str(summary_path), "--strict"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
