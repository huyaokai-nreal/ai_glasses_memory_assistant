"""Run the continuous E2E loop under one outer wall-clock timer.

Why this exists
---------------
``score_continuous_e2e.py`` reports ``full_loop_rtf`` as the sum of three
*in-process* stage RTFs (frontend + scoring + replay). That number is the
processing cost only: it starts after argument parsing in the scorer and after
probability loading in the frontend, so it excludes interpreter start-up,
module imports, input verification, model loading and artefact writing.

This wrapper measures what the real pipeline costs: it launches the two stages
back to back as subprocesses and times each one from process start to process
exit.

How to read the numbers (and how not to)
----------------------------------------
* Both stage durations come from **one** ``time.perf_counter`` clock inside this
  wrapper process. They are consecutive, non-overlapping spans, so
  ``frontend_process_seconds + scorer_process_seconds == full_process_seconds``
  holds by construction (the JSON carries ``stage_seconds_sum`` so the equality
  can be checked instead of assumed). Two independently measured, non-overlapping
  process durations may therefore be added.
* Human waiting between the stages is excluded by construction: this wrapper
  issues both commands itself with no pause in between, so only program work is
  on the clock.
* The timings are **not** to be confused with the durations a calling tool
  reports for the commands it ran. Those include shell/conda start-up, output
  capture and any human gap, and were never confirmed as process time; they are
  not evidence for an RTF claim. Only ``full_process_seconds`` from
  ``full-loop-timing.json`` is.

The result is written to ``full-loop-timing.json`` next to the other artefacts.
The RTF threshold stays at 1.0.

Usage
-----
    python run_full_loop_timed.py \
      --prep-dir reports/p3_continuous_prep/20260907-full8 \
      --predictions reports/p3_sortformer_continuous/20260907-full8-thr030 \
      --case-id R8001_M8004-full \
      --diar-variant micA_mean --activity-threshold 0.30 \
      --enhance-variant all8 --baseline-variant ch0 \
      --limit-seconds 75 \
      --source-run reports/eval_ali/eval-ali-e-candidate-fix2 \
      --out reports/p4_continuous_e2e/<new-run-dir> \
      --replay-service

Exit codes: 0 ok; 1 frontend stage failed (its own code is reported);
2 scorer stage failed; 3 loop finished but outer RTF exceeded the threshold;
4 the scorer produced no usable scores.json to read the duration from;
5 the output directory already exists (refusing to overwrite a previous run).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "continuous_full_loop_timing.v1"
RTF_MAX = 1.0
EXIT_OK = 0
EXIT_FRONTEND_FAILED = 1
EXIT_SCORER_FAILED = 2
EXIT_RTF_EXCEEDED = 3
EXIT_NO_DURATION = 4
EXIT_OUT_DIR_EXISTS = 5
#: The sub-command interpreter. Never sys.executable: the wrapper may run under
#: a different interpreter than the conda env that owns the stage, and using the
#: parent interpreter would silently bypass the env (wrong numpy/torch/sherpa).
STAGE_PYTHON = "python"


def find_conda(explicit: str | None) -> str:
    """Locate the conda executable without hardcoding any personal path.

    Order: explicit ``--conda`` -> ``$CONDA_EXE`` -> ``conda`` on PATH ->
    standard non-personal install roots (Homebrew/Anaconda system paths only).
    Anything else is an explicit error, never a guess.
    """
    if explicit:
        return explicit
    for candidate in (
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        "/opt/homebrew/anaconda3/bin/conda",
        "/opt/anaconda3/bin/conda",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("conda not found; pass --conda /path/to/conda")


def build_frontend_cmd(args: argparse.Namespace, out_dir: Path, repo_root: Path) -> list[str]:
    cmd = [
        STAGE_PYTHON,
        str(repo_root / "evals" / "audio_frontend" / "run_continuous_frontend.py"),
        "--prep-dir", str(Path(args.prep_dir).resolve()),
        "--predictions", str(Path(args.predictions).resolve()),
        "--case-id", args.case_id,
        "--diar-variant", args.diar_variant,
        "--activity-threshold", str(args.activity_threshold),
        "--enhance-variant", args.enhance_variant,
        "--baseline-variant", args.baseline_variant,
        "--out", str(out_dir / "frontend"),
    ]
    if args.limit_seconds and args.limit_seconds > 0:
        cmd += ["--limit-seconds", str(args.limit_seconds)]
    return cmd


def build_scorer_cmd(args: argparse.Namespace, out_dir: Path, repo_root: Path) -> list[str]:
    cmd = [
        STAGE_PYTHON,
        str(repo_root / "evals" / "audio_frontend" / "score_continuous_e2e.py"),
        "--prep-manifest", str(Path(args.prep_dir).resolve() / "prep-manifest.json"),
        "--frontend-run", str(out_dir / "frontend"),
        "--source-run", str(Path(args.source_run).resolve()),
        "--out", str(out_dir),
    ]
    if args.replay_service:
        cmd.append("--replay-service")
    if args.replay_timeout:
        cmd += ["--replay-timeout", str(args.replay_timeout)]
    if args.gate_scope and args.gate_scope != "auto":
        cmd += ["--gate-scope", args.gate_scope]
    return cmd


def run_stage(
    label: str,
    conda: str,
    env: str,
    cmd: list[str],
    runner=subprocess.run,
) -> tuple[int, str]:
    full = [conda, "run", "--no-capture-output", "-n", env] + cmd
    print(f"[{label}] $ {' '.join(full)}", flush=True)
    completed = runner(full, check=False)
    return int(completed.returncode), " ".join(full)


def _read_processed_seconds(out_dir: Path) -> tuple[float | None, str]:
    """Prefer scores.json; fall back to the frontend manifest."""
    scores_path = out_dir / "scores.json"
    if scores_path.is_file():
        try:
            return float(json.loads(scores_path.read_text(encoding="utf-8"))["processed_seconds"]), "scores.json"
        except (KeyError, TypeError, ValueError) as exc:
            print(f"could not read processed_seconds from {scores_path}: {exc}", file=sys.stderr)
    manifest_path = out_dir / "frontend" / "frontend-manifest.json"
    if manifest_path.is_file():
        try:
            return (
                float(json.loads(manifest_path.read_text(encoding="utf-8"))["case"]["processed_seconds"]),
                "frontend-manifest.json",
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"could not read processed_seconds from {manifest_path}: {exc}", file=sys.stderr)
    return None, ""


def main(argv: list[str] | None = None, *, runner=subprocess.run, clock=time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--diar-variant", default="micA_mean")
    parser.add_argument("--activity-threshold", type=float, default=0.30)
    parser.add_argument("--enhance-variant", default="all8")
    parser.add_argument("--baseline-variant", default="ch0")
    parser.add_argument("--limit-seconds", type=float, default=0.0, help="0 = full meeting")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay-service", action="store_true")
    parser.add_argument("--replay-timeout", type=float, default=30.0)
    parser.add_argument("--gate-scope", choices=("auto", "smoke", "full"), default="auto")
    parser.add_argument("--conda", default=None)
    parser.add_argument("--frontend-env", default="py311")
    parser.add_argument("--scorer-env", default="hermes")
    parser.add_argument("--repo-root", default=None, help="Defaults to the parent of this file's directory's parent.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = Path(args.out).resolve()
    # One run directory per attempt: never append to, overwrite or sit on top of
    # a previous run's artefacts (a fresh directory keeps the evidence clean).
    if out_dir.exists():
        print(
            f"output directory already exists, refusing to reuse it: {out_dir}\n"
            "choose a new --out directory for this attempt.",
            file=sys.stderr,
        )
        return EXIT_OUT_DIR_EXISTS
    conda = find_conda(args.conda)
    out_dir.mkdir(parents=True)

    frontend_cmd = build_frontend_cmd(args, out_dir, repo_root)
    scorer_cmd = build_scorer_cmd(args, out_dir, repo_root)

    # The clock starts the moment the first stage process is launched and stops
    # when the last one exits. No human step sits between the two subprocesses.
    loop_started = clock()
    frontend_code, frontend_display = run_stage("frontend", conda, args.frontend_env, frontend_cmd, runner=runner)
    frontend_seconds = clock() - loop_started
    scorer_code: int | None = None
    scorer_seconds = 0.0
    scorer_display = ""
    if frontend_code == 0:
        scorer_started = clock()
        scorer_code, scorer_display = run_stage("scorer", conda, args.scorer_env, scorer_cmd, runner=runner)
        scorer_seconds = clock() - scorer_started
        if scorer_code != 0:
            print(f"scorer stage failed with exit code {scorer_code}", file=sys.stderr)
    else:
        print(f"frontend stage failed with exit code {frontend_code}; aborting before scoring", file=sys.stderr)
    loop_seconds = clock() - loop_started

    processed_seconds, duration_source = _read_processed_seconds(out_dir)
    full_process_rtf = (loop_seconds / processed_seconds) if processed_seconds else None
    rtf_passed = full_process_rtf is not None and full_process_rtf <= RTF_MAX

    timing = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out_dir),
        "processed_seconds": processed_seconds,
        "processed_seconds_source": duration_source,
        "frontend_env": args.frontend_env,
        "scorer_env": args.scorer_env,
        "stage_python": STAGE_PYTHON,
        "frontend_process_seconds": round(frontend_seconds, 3),
        "scorer_process_seconds": round(scorer_seconds, 3),
        # The two stage spans are consecutive and share one clock, so they add up
        # to the loop total. Checked here rather than assumed.
        "stage_seconds_sum": round(frontend_seconds + scorer_seconds, 3),
        "full_process_seconds": round(loop_seconds, 3),
        "full_process_rtf": None if full_process_rtf is None else round(full_process_rtf, 6),
        "rtf_max": RTF_MAX,
        "full_process_rtf_passed": rtf_passed,
        "frontend_exit_code": frontend_code,
        "scorer_exit_code": scorer_code,
        "frontend_command": frontend_display,
        "scorer_command": scorer_display,
        "timing_basis": "single time.perf_counter inside run_full_loop_timed.py; the two stages run "
        "back to back as subprocesses, so their durations are consecutive, non-overlapping "
        "and additive (see stage_seconds_sum).",
        "excludes": "human waiting between stages (both stages are launched back to back by this wrapper); "
        "manual review steps. Includes process start-up, imports, input verification, "
        "probability/model loading, processing, replay and artefact writing.",
        "not_evidence": "durations reported by the calling tool/shell for these commands: those include "
        "shell and conda start-up, output capture and any human gap, and are not process timing.",
    }
    (out_dir / "full-loop-timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(timing, ensure_ascii=False, indent=2))

    if frontend_code != 0:
        return EXIT_FRONTEND_FAILED
    if scorer_code != 0:
        return EXIT_SCORER_FAILED
    if processed_seconds is None:
        return EXIT_NO_DURATION
    if not rtf_passed:
        print(
            f"outer loop RTF {full_process_rtf:.4f} exceeds {RTF_MAX}; real-time claim not supported",
            file=sys.stderr,
        )
        return EXIT_RTF_EXCEEDED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
