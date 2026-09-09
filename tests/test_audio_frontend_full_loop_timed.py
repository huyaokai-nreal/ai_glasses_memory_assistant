"""Unit tests for the outer full-loop timing wrapper.

No conda, no audio and no ASR are involved: the subprocess runner and the clock
are injected, so these tests only exercise command construction, the exit-code
contract and the timing arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "evals" / "audio_frontend", ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_full_loop_timed as wrapper  # noqa: E402

CONDA = "/bin/echo"  # never executed: the runner is injected


class FakeRunner:
    """Stands in for subprocess.run and records every command it was handed."""

    def __init__(self, returncodes: list[int], on_call=None) -> None:
        self.returncodes = list(returncodes)
        self.on_call = on_call
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], check: bool = False):
        index = len(self.calls)
        self.calls.append(list(cmd))
        if self.on_call:
            self.on_call(index, cmd)
        code = self.returncodes[index] if index < len(self.returncodes) else 0
        return type("Completed", (), {"returncode": code})()


class FakeClock:
    """Returns pre-baked monotonic readings in order."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.reads = 0

    def __call__(self) -> float:
        value = self.values[min(self.reads, len(self.values) - 1)]
        self.reads += 1
        return value


def _namespace(**overrides):
    import argparse

    defaults = dict(
        prep_dir="reports/p3_continuous_prep/20260907-full8",
        predictions="reports/p3_sortformer_continuous/20260907-full8-thr030",
        case_id="R8001_M8004-full",
        diar_variant="micA_mean",
        activity_threshold=0.30,
        enhance_variant="all8",
        baseline_variant="ch0",
        limit_seconds=75.0,
        source_run="reports/eval_ali/eval-ali-e-candidate-fix2",
        replay_service=True,
        replay_timeout=30.0,
        gate_scope="auto",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _argv(tmp_path: Path, **overrides) -> list[str]:
    out = tmp_path / "run"
    base = [
        "--prep-dir", "reports/p3_continuous_prep/20260907-full8",
        "--predictions", "reports/p3_sortformer_continuous/20260907-full8-thr030",
        "--case-id", "R8001_M8004-full",
        "--limit-seconds", "75",
        "--source-run", "reports/eval_ali/eval-ali-e-candidate-fix2",
        "--conda", CONDA,
        "--out", str(out),
        "--replay-service",
    ]
    for key, value in overrides.items():
        base += [f"--{key.replace('_', '-')}", str(value)]
    return base


def test_stage_commands_use_env_python_not_the_parent_interpreter() -> None:
    args = _namespace()
    out_dir = Path("/tmp/whatever")
    repo_root = Path("/repo")
    for cmd in (
        wrapper.build_frontend_cmd(args, out_dir, repo_root),
        wrapper.build_scorer_cmd(args, out_dir, repo_root),
    ):
        assert cmd[0] == "python"
        assert sys.executable not in cmd
        assert not any(part.startswith("/") and part.endswith("python") for part in cmd[:1])


def test_stage_commands_carry_the_locked_arguments() -> None:
    args = _namespace()
    out_dir = Path("/tmp/out")
    frontend = wrapper.build_frontend_cmd(args, out_dir, Path("/repo"))
    assert "--diar-variant" in frontend and "micA_mean" in frontend
    assert "--activity-threshold" in frontend and "0.3" in frontend
    assert "--enhance-variant" in frontend and "all8" in frontend
    assert "--baseline-variant" in frontend and "ch0" in frontend
    assert "--limit-seconds" in frontend and "75.0" in frontend
    scorer = wrapper.build_scorer_cmd(args, out_dir, Path("/repo"))
    assert "--replay-service" in scorer
    assert "--gate-scope" not in scorer  # auto is not forwarded


def test_scorer_command_forwards_explicit_scope() -> None:
    args = _namespace(gate_scope="full")
    scorer = wrapper.build_scorer_cmd(args, Path("/tmp/out"), Path("/repo"))
    assert scorer[scorer.index("--gate-scope") + 1] == "full"


def test_run_stage_prefixes_conda_run() -> None:
    runner = FakeRunner([0])
    code, display = wrapper.run_stage("frontend", CONDA, "py311", ["python", "x.py"], runner=runner)
    assert code == 0
    assert runner.calls[0][:5] == [CONDA, "run", "--no-capture-output", "-n", "py311"]
    assert runner.calls[0][5:] == ["python", "x.py"]
    assert display.startswith(f"{CONDA} run --no-capture-output -n py311 python x.py")


def test_existing_output_directory_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    runner = FakeRunner([])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=FakeClock([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert code == wrapper.EXIT_OUT_DIR_EXISTS
    assert runner.calls == []  # nothing was launched
    assert not (out / "full-loop-timing.json").exists()


def test_frontend_failure_skips_scorer(tmp_path: Path) -> None:
    runner = FakeRunner([1])
    clock = FakeClock([0.0, 10.0, 20.0])  # loop_started, after frontend, loop total
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_FRONTEND_FAILED
    assert len(runner.calls) == 1
    assert "run_continuous_frontend.py" in " ".join(runner.calls[0])
    timing = json.loads((tmp_path / "run" / "full-loop-timing.json").read_text(encoding="utf-8"))
    assert timing["frontend_exit_code"] == 1
    assert timing["scorer_exit_code"] is None


def test_scorer_failure_is_non_zero(tmp_path: Path) -> None:
    runner = FakeRunner([0, 2])
    clock = FakeClock([0.0, 10.0, 10.0, 25.0, 25.0])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_SCORER_FAILED
    assert len(runner.calls) == 2
    assert "score_continuous_e2e.py" in " ".join(runner.calls[1])


def test_missing_duration_is_non_zero(tmp_path: Path) -> None:
    runner = FakeRunner([0, 0])
    clock = FakeClock([0.0, 10.0, 10.0, 25.0, 25.0])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_NO_DURATION


def test_rtf_over_threshold_is_non_zero(tmp_path: Path) -> None:
    def write_scores(index: int, cmd: list[str]) -> None:
        if "score_continuous_e2e.py" in " ".join(cmd):
            out_index = cmd.index("--out") + 1
            Path(cmd[out_index]).mkdir(parents=True, exist_ok=True)
            (Path(cmd[out_index]) / "scores.json").write_text(
                json.dumps({"processed_seconds": 75.0}), encoding="utf-8"
            )

    runner = FakeRunner([0, 0], on_call=write_scores)
    # 100 s of process time for 75 s of audio => RTF 1.33 > 1.0
    clock = FakeClock([0.0, 50.0, 50.0, 100.0, 100.0])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_RTF_EXCEEDED
    timing = json.loads((tmp_path / "run" / "full-loop-timing.json").read_text(encoding="utf-8"))
    assert timing["full_process_rtf"] == pytest.approx(100.0 / 75.0)
    assert timing["full_process_rtf_passed"] is False


def test_success_path_writes_timing(tmp_path: Path) -> None:
    def write_scores(index: int, cmd: list[str]) -> None:
        if "score_continuous_e2e.py" in " ".join(cmd):
            out_index = cmd.index("--out") + 1
            Path(cmd[out_index]).mkdir(parents=True, exist_ok=True)
            (Path(cmd[out_index]) / "scores.json").write_text(
                json.dumps({"processed_seconds": 75.0}), encoding="utf-8"
            )

    runner = FakeRunner([0, 0], on_call=write_scores)
    # frontend 10 s, scorer 15 s, loop 25 s: the two spans add up by construction.
    clock = FakeClock([0.0, 10.0, 10.0, 25.0, 25.0])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_OK
    timing = json.loads((tmp_path / "run" / "full-loop-timing.json").read_text(encoding="utf-8"))
    assert timing["frontend_process_seconds"] == pytest.approx(10.0)
    assert timing["scorer_process_seconds"] == pytest.approx(15.0)
    assert timing["full_process_seconds"] == pytest.approx(25.0)
    assert timing["stage_seconds_sum"] == pytest.approx(timing["full_process_seconds"])
    assert timing["full_process_rtf"] == pytest.approx(25.0 / 75.0)
    assert timing["full_process_rtf_passed"] is True
    assert timing["stage_python"] == "python"
    assert timing["frontend_command"].startswith(f"{CONDA} run --no-capture-output -n py311 python")
    assert timing["scorer_command"].startswith(f"{CONDA} run --no-capture-output -n hermes python")


def test_success_path_uses_frontend_manifest_when_scores_missing(tmp_path: Path) -> None:
    def write_manifest(index: int, cmd: list[str]) -> None:
        if "run_continuous_frontend.py" in " ".join(cmd):
            out_index = cmd.index("--out") + 1
            Path(cmd[out_index]).mkdir(parents=True, exist_ok=True)
            (Path(cmd[out_index]) / "frontend-manifest.json").write_text(
                json.dumps({"case": {"processed_seconds": 75.0}}), encoding="utf-8"
            )

    runner = FakeRunner([0, 0], on_call=write_manifest)
    clock = FakeClock([0.0, 10.0, 10.0, 25.0, 25.0])
    code = wrapper.main(_argv(tmp_path), runner=runner, clock=clock)
    assert code == wrapper.EXIT_OK
    timing = json.loads((tmp_path / "run" / "full-loop-timing.json").read_text(encoding="utf-8"))
    assert timing["processed_seconds_source"] == "frontend-manifest.json"


def test_find_conda_has_no_personal_paths(monkeypatch) -> None:
    """find_conda() must never fall back to a personal absolute path (hygiene).
    With --conda missing and the standard env vars unset it either resolves a
    system path (fine) or raises — but it must NEVER return a path under a
    personal home directory."""
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    import shutil as _shutil

    real_which = _shutil.which

    def fake_which(name):
        if name == "conda":
            return None
        return real_which(name)

    monkeypatch.setattr(_shutil, "which", fake_which)
    try:
        resolved = wrapper.find_conda(None)
    except SystemExit:
        resolved = None  # acceptable: explicit failure beats a personal guess
    if resolved:
        assert "/Users/" not in resolved and "huyaokai" not in resolved
