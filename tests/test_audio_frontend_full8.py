"""Unit tests for the full8 batch driver + summarizer.

All tests are mock / fixture based: NO real audio, NO ASR, NO network. Fixtures live
in ``tests/fixtures/full8/`` (minimal, desensitized, committable — no absolute paths,
no real transcripts). The driver's subprocess runner and git inspector are injected so
nothing is actually executed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "full8"
AUDIO_FRONTEND = HERE.parent / "evals" / "audio_frontend"
sys.path.insert(0, str(AUDIO_FRONTEND))
sys.path.insert(0, str(FIX))  # for _gen_fixtures reuse

import _gen_fixtures as gen  # noqa: E402
import run_continuous_full8 as driver  # noqa: E402
import summarize_continuous_full8 as summ  # noqa: E402


def _load(name):
    d = FIX / name
    return (
        json.loads((d / "scores.json").read_text(encoding="utf-8")),
        json.loads((d / "full-loop-timing.json").read_text(encoding="utf-8")),
    )


def _patch_case_id(scores: dict, case_id: str) -> dict:
    """The summarizer now checks scores.case_id == manifest case_id (evidence
    failure on mismatch), so any fixture materialized for a case must carry that
    case's id."""
    scores = dict(scores)
    scores["case_id"] = case_id
    return scores


def _invokes(cmd, script_name):
    """True when ``cmd`` runs ``script_name``.

    The driver passes an ABSOLUTE path (``<repo>/evals/audio_frontend/<script>``),
    so a plain ``script_name in cmd`` list-membership check never matches. Match on
    the path suffix instead.
    """
    return any(str(c).endswith(script_name) for c in cmd)


def _materialize(cmd, fixture_name, case_id=None):
    """Write a fixture's scores.json/timing into the wrapper's --out dir."""
    if case_id is None:
        # The wrapper command always carries --case-id; patch the materialized
        # scores to that case so the summarizer's case_id==scores.case_id check
        # (an evidence failure on mismatch) stays meaningful.
        case_id = cmd[cmd.index("--case-id") + 1]
    out = Path(cmd[cmd.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "frontend").mkdir(exist_ok=True)
    s, t = _load(fixture_name)
    if case_id:
        s = _patch_case_id(s, case_id)
    (out / "scores.json").write_text(json.dumps(s), encoding="utf-8")
    (out / "full-loop-timing.json").write_text(json.dumps(t), encoding="utf-8")


def _copy_fixture(dst_root, case_id, fixture_name):
    """Copy a fixture's products into ``dst_root`` with the case id patched in."""
    dst = dst_root / case_id / "attempt0"
    dst.mkdir(parents=True, exist_ok=True)
    s, t = _load(fixture_name)
    (dst / "scores.json").write_text(json.dumps(_patch_case_id(s, case_id)), encoding="utf-8")
    (dst / "full-loop-timing.json").write_text(json.dumps(t), encoding="utf-8")
    return dst


# ---------------------------------------------------------------------------
# 1. command-contract: wrapper gets --prep-dir (never --prep-manifest), plus
#    --source-run / --predictions / --case-id
# ---------------------------------------------------------------------------
def test_wrapper_command_contract():
    cmd = driver.build_wrapper_command(
        conda="conda", wrapper_env="hermes", repo_root=AUDIO_FRONTEND.parent.parent,
        prep_dir=Path("P"), predictions=Path("Q"), source_run=Path("S"),
        case_id="R8001_M8004-full", attempt_dir=Path("OUT"),
        frontend_env="py311", scorer_env="hermes", replay_timeout=60.0,
    )
    joined = " ".join(cmd)
    assert "--prep-dir" in cmd and "--prep-manifest" not in cmd
    assert "--source-run" in cmd and "S" in cmd
    assert "--predictions" in cmd and "Q" in cmd
    assert "--case-id" in cmd and "R8001_M8004-full" in cmd
    assert "--gate-scope" in cmd and "full" in cmd
    assert "--out" in cmd and "OUT" in cmd
    assert "--replay-service" in cmd
    assert "--limit-seconds" not in joined  # full meeting


# ---------------------------------------------------------------------------
# 2. real JSON fixture path resolution (no drift from §8 field paths)
# ---------------------------------------------------------------------------
def test_fixture_field_paths():
    scores, timing = _load("session_pass")
    assert "checks" in scores["gates"]
    assert scores["variants"]["all8"]["cpcer"]["errors"] == 245
    assert scores["variants"]["all8"]["intervals"]["overlap"]["normalized_cer"] == 0.15
    assert scores["variants"]["all8"]["intervals"]["non_overlap"]["reference_chars"] == 9000
    assert timing["full_process_rtf_passed"] is True
    # diarization: all8 == ch0 (asserted before aggregation)
    a, c = scores["variants"]["all8"]["diarization"], scores["variants"]["ch0"]["diarization"]
    summ.assert_diarization_equal(a, c)


def test_classify_pass_is_evidence_valid():
    scores, timing = _load("session_pass")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None)
    assert st["evidence_valid"] and st["quality_passed"] and st["selected"]
    assert st["failure_class"] == "none"


# ---------------------------------------------------------------------------
# 3. 8-session integrity: manifest has exactly 8, all with selected_attempt_dir
# ---------------------------------------------------------------------------
def _make_batch_root_with_sessions(tmp_path, n=8, kind="session_pass"):
    root = tmp_path / "batch"
    for i in range(n):
        cid = f"CASE{i}"
        _copy_fixture(root, cid, kind)
    manifest = {
        "plan_id": "2026-09-08-audio-continuous-e2e-full8",
        "cases": [{"order": i, "case_id": f"CASE{i}"} for i in range(n)],
        "attempts": {
            f"CASE{i}": {"attempts": [{"attempt": 0, "attempt_dir": str(Path(f'CASE{i}') / 'attempt0'),
                                      "verifier_passed": True}],
                        "selected_attempt_dir": str(Path(f"CASE{i}") / "attempt0")}
            for i in range(n)
        },
    }
    (root / "batch-run-manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def test_eight_session_integrity(tmp_path):
    root, manifest = _make_batch_root_with_sessions(tmp_path, n=8)
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert len(sessions) == 8
    assert errors == []
    assert all(per_case[c]["status"]["selected"] for c in per_case)


# ---------------------------------------------------------------------------
# 4. weighted aggregation = Σnum / Σden (NOT average of percentages)
# ---------------------------------------------------------------------------
def test_weighted_aggregation_not_average():
    # two sessions: s1 all8 cpcer 100/1000=0.10 ; s2 all8 cpcer 1/10=0.10
    # average-of-percentages would be 0.10 too, so use skewed denominators:
    s1 = gen.build_scores(all8=gen._variant((100, 1000), (0, 2000), (0, 9000)),
                          ch0=gen._variant((200, 1000), (0, 2000), (0, 9000)),
                          dia=gen._diarization(100, 1000, 100, 1000, 0.10, 0.80))
    s2 = gen.build_scores(all8=gen._variant((1, 10), (0, 2000), (0, 9000)),
                          ch0=gen._variant((2, 10), (0, 2000), (0, 9000)),
                          dia=gen._diarization(1, 10, 1, 10, 0.10, 0.80))
    sess = [{"case_id": "A", "metrics": summ._session_metrics(s1),
             "internal_rtf_passed": True, "outer_rtf_passed": True},
            {"case_id": "B", "metrics": summ._session_metrics(s2),
             "result_gates_passed": True,
             "internal_rtf_passed": True, "outer_rtf_passed": True}]
    agg = summ.aggregate_sessions(sess)
    # pooled: (100+1)/(1000+10) = 101/1010 = 0.10 ; average of 0.10 and 0.10 = 0.10
    # use a clearly skewed example: 1/1000=0.001 vs 100/100=1.0 avg=0.5 but pooled=101/1100=0.0918
    s3 = gen.build_scores(all8=gen._variant((1, 1000), (0, 2000), (0, 9000)),
                          ch0=gen._variant((2, 1000), (0, 2000), (0, 9000)),
                          dia=gen._diarization(1, 1000, 1, 1000, 0.001, 0.80))
    s4 = gen.build_scores(all8=gen._variant((100, 100), (0, 2000), (0, 9000)),
                          ch0=gen._variant((200, 100), (0, 2000), (0, 9000)),
                          dia=gen._diarization(100, 100, 100, 100, 1.0, 0.80))
    sess2 = [{"case_id": "C", "metrics": summ._session_metrics(s3),
              "internal_rtf_passed": True, "outer_rtf_passed": True},
             {"case_id": "D", "metrics": summ._session_metrics(s4),
              "internal_rtf_passed": True, "outer_rtf_passed": True}]
    agg2 = summ.aggregate_sessions(sess2)
    pooled = agg2["variants"]["all8"]["cpcer"]["rate"]
    avg_pct = (1.0 + 0.001) / 2
    assert abs(pooled - (101 / 1100)) < 1e-9
    assert abs(pooled - avg_pct) > 0.4  # clearly different from naive average


# ---------------------------------------------------------------------------
# 5. three low-recall sessions but aggregate passes; worst-case ranking flags them
# ---------------------------------------------------------------------------
def test_three_low_recall_aggregate_passes():
    sessions = []
    # 5 high-recall sessions
    for i in range(5):
        s = gen.build_scores(
            all8=gen._variant((245, 1000), (300, 2000), (200, 9000)),
            ch0=gen._variant((438, 1000), (400, 2000), (300, 9000)),
            dia=gen._diarization(4727, 25607, 4000, 5000, 0.1846, 0.80))
        sessions.append({"case_id": f"H{i}", "metrics": summ._session_metrics(s),
                         "result_gates_passed": True,
                         "internal_rtf_passed": True, "outer_rtf_passed": True})
    # 3 low-recall sessions with the known sub-70% recall values. Frame counts are
    # chosen so det/ref reproduces the exact decimal (snap-free, fixture-consistent).
    lows = [("L0", 2672, 4000), ("L1", 3492, 5000), ("L2", 3050, 4693)]
    for cid, det, ref in lows:
        s = gen.build_scores(
            all8=gen._variant((245, 1000), (300, 2000), (200, 9000)),
            ch0=gen._variant((438, 1000), (400, 2000), (300, 9000)),
            dia=gen._diarization(4727, 25607, det, ref, 0.1846, None))
        sessions.append({"case_id": cid, "metrics": summ._session_metrics(s),
                         "result_gates_passed": True,
                         "internal_rtf_passed": True, "outer_rtf_passed": True})
    agg = summ.aggregate_sessions(sessions)
    gates = summ.evaluate_summary_gates(agg, sessions, all_evidence_valid=True)
    # recall gate must pass (aggregate >= 0.70)
    assert gates["gates"]["aggregate_overlap_recall_ge_70"] is True
    # DER/recall are aggregate-only: the 3 sub-70% sessions are NOT labelled
    # quality_fail (their per-session result gates are all green)
    assert gates["gates"]["all_session_result_gates_passed"] is True
    ranking = summ.worst_case_ranking(sessions)
    # the 3 low sessions must be listed as the three worst for recall (worst first)
    worst = ranking["recall"]["worst"]
    assert worst is not None
    assert worst[0] == "L2"  # 3050/4693 ≈ 0.6499 is the single worst
    top3 = [cid for cid, _ in ranking["recall"]["ranked_worst_first"][:3]]
    assert set(top3) == {"L0", "L1", "L2"}
    # each low session individually < 0.70 and the rates are the exact fractions
    for cid, det, ref in lows:
        rate = sessions[[x["case_id"] for x in sessions].index(cid)]["metrics"]["variants"]["all8"]["recall"]["rate"]
        assert rate == pytest.approx(det / ref)
        assert rate < 0.70
    assert ranking["recall"]["n_evaluable"] == 8
    assert ranking["recall"]["not_evaluable_cases"] == []
    # the aggregate is the weighted Σdet/Σref over all 8 sessions and must be >= 0.70
    total_det = 5 * 4000 + sum(d for _, d, _ in lows)
    total_ref = 5 * 5000 + sum(r for _, _, r in lows)
    assert agg["variants"]["all8"]["recall"]["rate"] == pytest.approx(total_det / total_ref)


# ---------------------------------------------------------------------------
# 6. quality failure continues (still selected), batch ends red
# ---------------------------------------------------------------------------
def test_quality_fail_still_selected():
    """Per-session result failure (cpCER regression) keeps the attempt selected."""
    scores, timing = _load("session_quality_fail")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None)
    assert st["evidence_valid"] is True
    assert st["quality_passed"] is False
    assert st["selected"] is True
    assert st["failure_class"] == "quality_fail"
    # DER is NOT a per-session check (aggregate-only); the failure is cpCER regression
    assert any("cpcer_strictly_better" in r for r in st["failure_reasons"])
    assert not any("der_le_25" in r for r in st["failure_reasons"])


# ---------------------------------------------------------------------------
# 7. evidence failure aborts (not selected); product missing -> infra_fail
# ---------------------------------------------------------------------------
def test_evidence_fail_not_selected():
    scores, timing = _load("session_evidence_fail")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None)
    assert st["evidence_valid"] is False
    assert st["selected"] is False
    assert st["failure_class"] == "evidence_fail"


def test_product_missing_is_infra_fail():
    st = summ.classify_missing_product("scores.json absent")
    assert st["evidence_valid"] is False and st["selected"] is False
    assert st["failure_class"] == "infra_fail"


def test_collect_selected_rejects_evidence_fail(tmp_path):
    root = tmp_path / "batch"
    _copy_fixture(root, "CASE0", "session_evidence_fail")
    manifest = {
        "plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
        "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0", "verifier_passed": True}],
                               "selected_attempt_dir": "CASE0/attempt0"}},
    }
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert sessions == []  # not selected into aggregation
    assert any("CASE0" in e for e in errors)
    assert per_case["CASE0"]["status"]["failure_class"] == "evidence_fail"


# ---------------------------------------------------------------------------
# 8. retry selection: only transient launch-exception / timeout retries
# ---------------------------------------------------------------------------
def test_decide_transient_only_for_infra():
    assert driver.decide_transient("launch_exception") is True
    assert driver.decide_transient("timeout") is True
    assert driver.decide_transient("completed") is False


class FakeRunner:
    """Mimics subprocess.run; optionally materializes scores for wrapper calls."""

    def __init__(self, raise_on_wrapper_call=0, case_id=None):
        self.calls = []
        self.raise_remaining = raise_on_wrapper_call
        self.case_id = case_id

    def __call__(self, cmd, timeout=None):
        self.calls.append((list(cmd), timeout))
        if _invokes(cmd, "run_full_loop_timed.py"):
            if self.raise_remaining > 0:
                self.raise_remaining -= 1
                raise FileNotFoundError("conda missing")
            # materialize a passing scores.json + timing in the attempt dir; the
            # scores case_id defaults to the case the wrapper was launched for
            _materialize(cmd, "session_pass", case_id=self.case_id)
            return SimpleNamespace(returncode=0)
        if _invokes(cmd, "verify_frontend_wavs.py"):
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)


def test_process_case_retries_transient_then_selects(tmp_path):
    root = tmp_path / "batch"
    runner = FakeRunner(raise_on_wrapper_call=1, case_id="R1")  # attempt0 launch fails, attempt1 ok
    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=root, conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=runner, dry_run=False,
    )
    assert not aborted, reason
    assert sel is not None
    assert sel == str(Path("R1") / "attempt1")  # relative path persisted in the manifest
    assert len(recs) == 2  # attempt0 failed transient, attempt1 succeeded
    assert recs[0]["status"]["failure_class"] == "infra_fail"
    assert recs[1]["status"]["selected"] is True
    # attempt records carry the RELATIVE dir (exact match for the summarizer)
    assert recs[1]["attempt_dir"] == "R1/attempt1"
    # the retry really re-ran the wrapper (2 wrapper launches, 1 verifier on success)
    wrapper_calls = [c for c, _ in runner.calls if _invokes(c, "run_full_loop_timed.py")]
    verifier_calls = [c for c, _ in runner.calls if _invokes(c, "verify_frontend_wavs.py")]
    assert len(wrapper_calls) == 2
    assert len(verifier_calls) == 1


def test_process_case_quality_fail_no_retry(tmp_path):
    root = tmp_path / "batch"
    runner = FakeRunner(raise_on_wrapper_call=0)

    # override materialization to write a quality_fail scores instead
    def quality_runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_quality_fail")
            # scorer returns 2 when gates.passed is False; scores.json still written
            return SimpleNamespace(returncode=2)
        return SimpleNamespace(returncode=0)

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=root, conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=quality_runner, dry_run=False,
    )
    assert not aborted, reason
    assert sel is not None
    assert len(recs) == 1  # no retry for quality failure
    assert recs[0]["status"]["failure_class"] == "quality_fail"
    assert recs[0]["status"]["selected"] is True
    # the verifier MUST still have run even though the wrapper exited 2 (quality fail)
    assert recs[0]["verifier_exit"] == 0
    assert recs[0]["verifier_passed"] is True


def test_verifier_runs_even_when_wrapper_exit_nonzero(tmp_path):
    """A quality failure exits the wrapper 2; WAV evidence must still be verified."""
    calls = []

    def runner(cmd, timeout=None):
        calls.append(list(cmd))
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_quality_fail")
            return SimpleNamespace(returncode=2)
        return SimpleNamespace(returncode=0)

    driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=runner, dry_run=False,
    )
    assert any(_invokes(c, "verify_frontend_wavs.py") for c in calls)


def test_unverified_wavs_are_evidence_fail(tmp_path):
    """Verifier launch failure => evidence UNVERIFIED => fail-closed, never assumed ok."""
    def runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_pass")
            return SimpleNamespace(returncode=0)
        raise FileNotFoundError("verifier cannot launch")

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=runner, dry_run=False,
    )
    assert sel is None
    assert aborted is True
    assert recs[0]["status"]["evidence_valid"] is False
    assert any("verifier_failed" in r for r in recs[0]["status"]["failure_reasons"])


def test_collect_selected_fail_closed_without_verifier_record(tmp_path):
    """No recorded verifier result for the selected dir => not aggregated."""
    root = tmp_path / "batch"
    _copy_fixture(root, "CASE0", "session_pass")
    manifest = {
        "plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
        # attempt record present but WITHOUT verifier_passed
        "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0"}],
                               "selected_attempt_dir": "CASE0/attempt0"}},
    }
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert sessions == []
    assert per_case["CASE0"]["status"]["evidence_valid"] is False
    assert errors


def test_process_case_evidence_fail_aborts_no_retry(tmp_path):
    root = tmp_path / "batch"
    runner = FakeRunner(raise_on_wrapper_call=0)

    def evidence_runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_evidence_fail")
            return SimpleNamespace(returncode=2)
        return SimpleNamespace(returncode=0)

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=root, conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=evidence_runner, dry_run=False,
    )
    assert aborted is True
    assert sel is None
    assert len(recs) == 1  # evidence failure never retried
    assert recs[0]["status"]["failure_class"] == "evidence_fail"
    assert recs[0]["status"]["selected"] is False
    assert any("network_calls_zero" in r for r in recs[0]["status"]["failure_reasons"])
    assert reason and "evidence_fail" in reason


# ---------------------------------------------------------------------------
# 9. manifest must not mix in failed / old / retry dirs (only selected_attempt_dir)
# ---------------------------------------------------------------------------
def test_interrupt_mid_session_is_not_evidence_fail(tmp_path):
    """A signal that kills the wrapper must read as `interrupted`, never evidence_fail,
    and must NOT be retried."""
    stopped = {"flag": False}

    def runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            stopped["flag"] = True  # simulate SIGINT reaching the process group
            return SimpleNamespace(returncode=-2)
        return SimpleNamespace(returncode=0)

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=runner, dry_run=False,
        should_stop=lambda: stopped["flag"],
    )
    assert aborted is True
    assert sel is None
    assert len(recs) == 1  # never auto-restarted
    assert recs[0]["status"]["failure_class"] == "interrupted"
    assert "interrupted" in reason


def test_interrupt_before_attempt_stops_without_running(tmp_path):
    def boom(cmd, timeout=None):
        raise AssertionError("must not run after interrupt")

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=boom, dry_run=False,
        should_stop=lambda: True,
    )
    assert aborted is True and sel is None and recs == []
    assert "interrupted before attempt0" in reason


def test_timeout_is_transient_and_retried(tmp_path):
    """A pre-defined stage timeout is the only other retryable class."""
    import subprocess as _sp
    state = {"n": 0}

    def runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            state["n"] += 1
            if state["n"] == 1:
                raise _sp.TimeoutExpired(cmd, timeout or 1)
            _materialize(cmd, "session_pass")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)

    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=5.0, runner=runner, dry_run=False,
    )
    assert not aborted, reason
    assert sel is not None
    assert len(recs) == 2
    assert recs[0]["status"]["failure_class"] == "infra_fail"


def test_retry_quota_exhausted_aborts(tmp_path):
    """retry1/retry2 are a max quota, not a requirement: 3 transients => abort."""
    runner = FakeRunner(raise_on_wrapper_call=3)  # attempt0,1,2 all fail to launch
    sel, recs, aborted, reason = driver.process_case(
        "R1", batch_root=tmp_path / "batch", conda="conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, frontend_env="py311", scorer_env="hermes",
        replay_timeout=60.0, expected_hashes=None, max_retries=2,
        stage_timeout=None, runner=runner, dry_run=False,
    )
    assert aborted is True and sel is None
    assert len(recs) == 3  # attempt0 + retry1 + retry2, then stop
    assert "exhausted retries" in reason


def test_manifest_only_reads_selected(tmp_path):
    root = tmp_path / "batch"
    _copy_fixture(root, "CASE0", "session_pass")
    # a stray failed dir that must NOT be read
    bad = root / "CASE0" / "attempt1"
    bad.mkdir(parents=True)
    s, t = _load("session_evidence_fail")
    (bad / "scores.json").write_text(json.dumps(_patch_case_id(s, "CASE0")), encoding="utf-8")
    (bad / "full-loop-timing.json").write_text(json.dumps(t), encoding="utf-8")
    manifest = {
        "plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
        "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0", "verifier_passed": True}],
                               "selected_attempt_dir": "CASE0/attempt0"}},  # points only to good
    }
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert len(sessions) == 1
    assert per_case["CASE0"]["status"]["evidence_valid"] is True


# ---------------------------------------------------------------------------
# 10. zero-denominator fail-closed: per-session not_evaluable; aggregate denom=0
#     fails; worst-case ranking skips not_evaluable
# ---------------------------------------------------------------------------
def test_zero_denominator_not_evaluable(tmp_path):
    root = tmp_path / "batch"
    _copy_fixture(root, "CASE0", "session_not_evaluable")
    manifest = {
        "plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
        "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0", "verifier_passed": True}],
                               "selected_attempt_dir": "CASE0/attempt0"}},
    }
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert len(sessions) == 1
    nov = sessions[0]["metrics"]["variants"]["all8"]["non_overlap"]
    assert nov["not_evaluable"] is True


def test_aggregate_zero_denominator_fails():
    # all 8 sessions have non_overlap reference_chars = 0 -> aggregate denom 0
    sessions = []
    for i in range(8):
        s = gen.build_scores(
            all8=gen._variant((245, 1000), (300, 2000), (0, 0)),
            ch0=gen._variant((438, 1000), (400, 2000), (0, 0)),
            dia=gen._diarization(4727, 25607, 4103, 5697, 0.1846, 0.7202))
        sessions.append({"case_id": f"S{i}", "metrics": summ._session_metrics(s),
                         "result_gates_passed": True,
                         "internal_rtf_passed": True, "outer_rtf_passed": True})
    agg = summ.aggregate_sessions(sessions)
    assert agg["variants"]["all8"]["non_overlap"]["aggregate_denominator_zero"] is True
    gates = summ.evaluate_summary_gates(agg, sessions, all_evidence_valid=True)
    # gate 4 (non-overlap <= 11.5%) must FAIL because its aggregate denominator is 0
    assert gates["gates"]["aggregate_non_overlap_cer_le_11_5"] is False
    # ... but the all-session result gate stays green (per-session CER never failed)
    assert gates["gates"]["all_session_result_gates_passed"] is True
    ranking = summ.worst_case_ranking(sessions)
    assert ranking["non_overlap_cer"]["n_not_evaluable"] == 8


# ---------------------------------------------------------------------------
# atomic manifest write preserves all old attempt records
# ---------------------------------------------------------------------------
def test_atomic_manifest_preserves_history(tmp_path):
    root = tmp_path / "batch"
    m = {"schema": "x", "attempts": {"C": {"attempts": [{"attempt": 0, "attempt_dir": "a0"}],
                                           "selected_attempt_dir": "a0"}}}
    driver.write_manifest_atomic(root, m)
    # simulate appending a retry attempt, then rewrite atomically
    m["attempts"]["C"]["attempts"].append({"attempt": 1, "attempt_dir": "a1"})
    m["attempts"]["C"]["selected_attempt_dir"] = "a1"
    driver.write_manifest_atomic(root, m)
    loaded = json.loads((root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    assert len(loaded["attempts"]["C"]["attempts"]) == 2  # both attempts preserved
    assert loaded["attempts"]["C"]["selected_attempt_dir"] == "a1"
    assert list(root.glob("batch-run-manifest.json.tmp.*")) == []  # temp cleaned


# ---------------------------------------------------------------------------
# diarization assertion fails on mismatch (evidence failure)
# ---------------------------------------------------------------------------
def test_diarization_assert_mismatch():
    a = {"errors": 1, "reference_speaker_frames": 10, "overlap_detected_frames": 5, "overlap_reference_frames": 7}
    c = {"errors": 2, "reference_speaker_frames": 10, "overlap_detected_frames": 5, "overlap_reference_frames": 7}
    with pytest.raises(ValueError):
        summ.assert_diarization_equal(a, c)


# ---------------------------------------------------------------------------
# dry-run invokes NOTHING (no subprocess runner called)
# ---------------------------------------------------------------------------
class FakeGit:
    def head_sha(self): return "abc123def456"
    def status_porcelain(self): return ""
    def runtime_sources_locked(self, files): return (True, [])
    def is_tracked(self, rel): return True
    def matches_head(self, rel): return True


class FakeEnvProbe:
    """Stand-in for EnvProbe: never runs conda, always resolves a python."""

    def __init__(self, py="/usr/bin/python3", ver="3.11.0"):
        self.py, self.ver = py, ver

    def __call__(self, conda, env):
        return self.py, self.ver


def _cli_args(tmp_path, prep_dir=None, extra=None):
    return [
        "--batch-root", str(tmp_path / "out"),
        "--prep-dir", str(prep_dir or FIX),
        "--predictions", str(FIX / "predictions"),
        "--source-run", str(FIX / "source-run"),
        "--conda", "conda-stub",
    ] + (extra or [])


#: Injected free disk so fixture tests never depend on the host's actual space.
_BIG_DISK = 5000.0


def test_dry_run_invokes_nothing(tmp_path, capsys):
    class BoomRunner:
        def __call__(self, cmd, timeout=None):
            raise AssertionError("runner must not be called in dry-run")

    rc = driver.main(_cli_args(tmp_path, extra=["--dry-run"]),
                     runner=BoomRunner(), git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "NO wrapper/verifier/ASR invoked" in out
    assert not (tmp_path / "out" / "batch-root-manifest.json").exists()
    assert not (tmp_path / "out" / "batch-run-manifest.json").exists()
    # the exact command pair for all eight cases is previewed (one wrapper + one
    # verifier line each). Count on markers that appear only in the command preview:
    # the script names also occur inside the manifest's runtime_source_sha256 block.
    assert out.count("attempt0 would run:") == 8
    assert out.count("--case-id") == 8
    assert out.count("--gate-scope full") == 8
    assert "8 cases planned" in out
    for cid in gen.CASES:
        assert cid in out
    # nothing was written into any attempt dir either
    assert not (tmp_path / "out" / gen.CASES[0]).exists()


# ---------------------------------------------------------------------------
# model-hash binding is read from the real manifests and never fails open
# ---------------------------------------------------------------------------
def test_expected_hashes_read_from_manifests():
    hashes, problems = driver.read_expected_hashes(FIX / "source-run", FIX / "predictions")
    assert problems == []
    assert hashes["asr_model_sha256"] == gen.PLACEHOLDER_ASR == driver.LOCKED_ASR_MODEL_SHA256
    assert hashes["sortformer_model_sha256"] == gen.PLACEHOLDER_SF == driver.LOCKED_SORTFORMER_MODEL_SHA256


def test_missing_manifest_is_preflight_failure(tmp_path):
    hashes, problems = driver.read_expected_hashes(tmp_path / "nope", tmp_path / "nope2")
    assert hashes == {}
    assert len(problems) == 2  # both manifests missing -> hard preflight failure


def test_self_certifying_hash_is_preflight_failure(tmp_path):
    """A manifest that declares a non-locked hash must be refused (self-certify ban)."""
    src = tmp_path / "source-run"
    pred = tmp_path / "predictions"
    src.mkdir(parents=True)
    pred.mkdir(parents=True)
    (src / "run-manifest.json").write_text(json.dumps({
        "runtime": {"ambient_audio_profile": {"asr_model_sha256": "DEADBEEF"}}}))
    (pred / "run-manifest.json").write_text(json.dumps({"model": {"sha256": "DEADBEEF"}}))
    hashes, problems = driver.read_expected_hashes(src, pred)
    assert hashes == {}
    assert len(problems) == 2  # both hashes != locked constant -> hard preflight failure


def test_hash_drift_is_evidence_fail():
    """The comparison target is the FIXED LOCKED constant: a session whose
    recorded hash differs from the lock is evidence_fail (self-certify ban)."""
    scores, timing = _load("session_pass")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes={},
                               case_id=scores["case_id"])
    # session_pass carries the real locked hash -> schema ok
    assert st["evidence_valid"] is True
    scores["asr_model_sha256"] = "A_DIFFERENT_SHA"
    st2 = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes={},
                                case_id=scores["case_id"])
    assert st2["evidence_valid"] is False
    assert any("asr_model_sha256" in r for r in st2["failure_reasons"])


def test_wrong_case_count_aborts(tmp_path):
    prep = tmp_path / "prep"
    prep.mkdir()
    seven = json.loads((FIX / "prep-manifest.json").read_text(encoding="utf-8"))
    seven["cases"] = seven["cases"][:7]
    (prep / "prep-manifest.json").write_text(json.dumps(seven), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        driver.main(_cli_args(tmp_path, prep_dir=prep, extra=["--dry-run"]),
                    runner=None, git=FakeGit(),
                    env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    msg = str(exc.value)
    assert "locked eight" in msg
    assert "expected" in msg


# ---------------------------------------------------------------------------
# end-to-end: one foreground command runs all eight sessions and summarizes
# ---------------------------------------------------------------------------
def test_driver_main_end_to_end_eight_sessions_green(tmp_path):
    runner = FakeRunner(case_id=None)  # per-case: materialize with the wrapper's --case-id
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0

    batch_root = tmp_path / "out"
    manifest = json.loads((batch_root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 8
    assert [c["case_id"] for c in manifest["cases"]] == list(gen.CASES)
    assert manifest["git"]["head_sha"] == "abc123def456"
    assert manifest["git"]["runtime_sources_match_head"] is True
    assert manifest["schema"] == driver.SCHEMA
    # env fields resolve to real python (never null) via the injected probe
    assert manifest["env"]["frontend_python"] == "/usr/bin/python3"
    assert manifest["env"]["frontend_python_version"] == "3.11.0"
    assert manifest["env"]["scorer_python_version"] == "3.11.0"
    # source-run manifest hash is recorded (drift evidence for the batch)
    assert manifest["locks"]["source_run_manifest_sha256"]
    # post_run populated at the end of the batch
    assert manifest["post_run"]["status"] == "completed"
    assert manifest["post_run"]["code_drift"] == []
    # every managed runtime source under the three locked roots is hashed into the
    # manifest (directory enumeration, not a hand-picked import list).
    src = manifest["git"]["runtime_source_sha256"]
    assert len(src) >= 80  # the three run-code roots enumerate to dozens of .py files
    for rel in ("evals/audio_frontend/run_continuous_full8.py",
                "evals/audio_frontend/enhance_mvdr_oracle.py",
                "ai_glasses_memory_assistant/agent_bridge.py",
                "scripts/p0_oracle_interval_ablation.py"):
        assert rel in src, rel
    assert "run_continuous_full8.py" in " ".join(src)
    assert "summarize_continuous_full8.py" in " ".join(src)
    # all eight cases selected; selected path is RELATIVE <case_id>/attempt0
    for case in manifest["cases"]:
        cid = case["case_id"]
        assert manifest["attempts"][cid]["selected_attempt_dir"] == f"{cid}/attempt0"
        rec = manifest["attempts"][cid]["attempts"][0]
        assert rec["attempt_dir"] == f"{cid}/attempt0"
        assert rec["attempt_dir"] == manifest["attempts"][cid]["selected_attempt_dir"]

    summary = json.loads((batch_root / "full8-summary.json").read_text(encoding="utf-8"))
    assert summary["n_selected"] == 8
    assert summary["errors"] == []
    assert summary["summary_gates"]["passed"] is True
    assert all(summary["summary_gates"]["gates"].values())
    assert summary["summary_gates"]["gates"]["all_session_result_gates_passed"] is True
    # aggregation really pooled all eight sessions (245 errors x 8 = 1960)
    assert summary["aggregate"]["variants"]["all8"]["cpcer"]["errors"] == 245 * 8
    assert summary["aggregate"]["variants"]["all8"]["cpcer"]["reference_chars"] == 1000 * 8


def test_driver_main_end_to_end_quality_fail_is_red_but_complete(tmp_path):
    """Quality failure (cpCER regression): all eight still selected, batch red."""
    def quality_runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_quality_fail")
            return SimpleNamespace(returncode=2)
        return SimpleNamespace(returncode=0)

    rc = driver.main(_cli_args(tmp_path), runner=quality_runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 1  # summarizer --strict

    batch_root = tmp_path / "out"
    summary = json.loads((batch_root / "full8-summary.json").read_text(encoding="utf-8"))
    assert summary["n_selected"] == 8  # evidence valid -> all selected
    assert summary["summary_gates"]["passed"] is False
    # the failing per-session result gate makes the batch red via the NEW gate,
    # even though the pooled aggregate still improves (0.50 < 0.438? no: 0.50 > 0.438,
    # so the aggregate gate is also red here — the assert below pins the new gate too)
    assert summary["summary_gates"]["gates"]["all_session_result_gates_passed"] is False
    assert summary["summary_gates"]["gates"]["aggregate_all8_cpcer_lt_ch0"] is False
    assert summary["summary_gates"]["gates"]["aggregate_der_le_25"] is True
    # evidence gate itself stays green: this is a result failure, not an evidence gap
    assert summary["summary_gates"]["gates"]["all_evidence_privacy_integrity_passed"] is True
    # aggregate DER unchanged from the pass fixture (per-session der was NOT the
    # failure driver): pooled 8x4727/8x25607
    assert summary["aggregate"]["variants"]["all8"]["der"]["rate"] == pytest.approx(4727 / 25607)


def test_driver_main_aborts_on_first_evidence_gap(tmp_path):
    """Evidence failure on session 1 stops the batch; later cases never run."""
    def evidence_runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_evidence_fail")
            return SimpleNamespace(returncode=2)
        return SimpleNamespace(returncode=0)

    rc = driver.main(_cli_args(tmp_path), runner=evidence_runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 2

    batch_root = tmp_path / "out"
    manifest = json.loads((batch_root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    # only the first case was attempted
    assert len(manifest["attempts"]) == 1
    first = manifest["cases"][0]["case_id"]
    assert manifest["attempts"][first]["selected_attempt_dir"] is None
    assert not (batch_root / "full8-summary.json").exists()


# ===========================================================================
# Round-2 additions (review feedback groups 1-6)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. selected_attempt_dir: relative path <case_id>/attemptN + boundary checks
# ---------------------------------------------------------------------------
def test_resolve_attempt_dir_relative_stays_in_batch(tmp_path):
    """A real relative --batch-root must resolve under that root, exactly once."""
    batch = tmp_path / "batch"
    attempt_dir, err = summ.resolve_attempt_dir(batch, "R8001_M8004-full/attempt0")
    assert err is None
    assert attempt_dir == (batch / "R8001_M8004-full" / "attempt0").resolve()
    # prefix confusion: an attempt dir that merely STARTS with the batch name is
    # not inside it (guards against naive string concatenation)
    outside = tmp_path / "batch-other" / "attempt0"
    outside.mkdir(parents=True)
    attempt_dir2, err2 = summ.resolve_attempt_dir(tmp_path / "batch", f"../batch-other/attempt0")
    assert err2 is not None and "escapes" in err2


def test_resolve_attempt_dir_rejects_escape_and_empty(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir(parents=True)
    assert summ.resolve_attempt_dir(batch, None)[1] is not None
    assert summ.resolve_attempt_dir(batch, "")[1] is not None
    assert summ.resolve_attempt_dir(batch, "../../etc/passwd")[1] is not None
    # symlink escape is defeated by resolve(): a link inside batch pointing at a
    # dir outside resolves outside -> rejected
    victim = tmp_path / "victim"
    victim.mkdir()
    link = batch / "evil"
    try:
        link.symlink_to(victim, target_is_directory=True)
    except OSError:
        pass  # symlink unsupported: fall through to the non-link checks above
    if link.exists():
        d, err = summ.resolve_attempt_dir(batch, "evil/attempt0")
        assert err is not None and "escapes" in err


def test_e2e_manifest_uses_relative_paths_no_double_join(tmp_path):
    """End-to-end with REAL relative --batch-root: manifest must carry
    <case_id>/attemptN (not an absolute path, not a doubled prefix)."""
    batch = tmp_path / "real-batch-root"
    runner = FakeRunner(case_id=None)
    rc = driver.main(["--batch-root", str(batch), "--prep-dir", str(FIX),
                      "--predictions", str(FIX / "predictions"),
                      "--source-run", str(FIX / "source-run"),
                      "--conda", "conda-stub"],
                     runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    manifest = json.loads((batch / "batch-run-manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        cid = case["case_id"]
        sel = manifest["attempts"][cid]["selected_attempt_dir"]
        assert sel == f"{cid}/attempt0"
        assert not Path(sel).is_absolute()
        # collect_selected re-resolves from a DIFFERENT cwd-style relative base
        # without doubling (the regression this guards: batch_root/sel twice)
        from pathlib import Path as P
        resolved, err = summ.resolve_attempt_dir(batch, sel)
        assert err is None
        assert (resolved / "scores.json").is_file()


# ---------------------------------------------------------------------------
# 2. gate semantics: DER/recall aggregate-only; per-session result gate added
# ---------------------------------------------------------------------------
def _session_dict(scores, cid="C"):
    return {"case_id": cid, "metrics": summ._session_metrics(scores),
            "result_gates_passed": True, "internal_rtf_passed": True, "outer_rtf_passed": True}


def test_quality_fail_reasons_never_include_der_recall():
    """DER/overlap recall must NEVER appear as a per-session failure reason."""
    scores, timing = _load("session_quality_fail")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None)
    for r in st["failure_reasons"]:
        assert "der_le_25" not in r and "overlap_recall_ge_70" not in r


def test_one_session_cpcer_regression_but_aggregate_improves_is_red():
    """Brain test: one session's relative cpCER fails while the pooled aggregate
    still improves -> batch MUST be red via all_session_result_gates_passed."""
    sessions = []
    # 7 healthy sessions: all8 cpCER 0.10 vs ch0 0.30
    for i in range(7):
        s = gen.build_scores(
            all8=gen._variant((100, 1000), (0, 2000), (0, 9000)),
            ch0=gen._variant((300, 1000), (0, 2000), (0, 9000)),
            dia=gen._diarization(100, 1000, 100, 1000, None, None))
        sessions.append(_session_dict(s, cid=f"OK{i}"))
    # 1 failing session: all8 cpCER 0.40 vs ch0 0.30 (regression), small weight
    s_bad = gen.build_scores(
        all8=gen._variant((4, 10), (0, 2000), (0, 9000)),
        ch0=gen._variant((3, 10), (0, 2000), (0, 9000)),
        dia=gen._diarization(1, 10, 1, 10, None, None),
        checks_overrides={"cpcer_strictly_better": False})
    bad_sess = _session_dict(s_bad, cid="BAD")
    bad_sess["result_gates_passed"] = False  # cpCER regression -> session gate red
    sessions.append(bad_sess)
    # aggregate cpCER: (700+4)/(7000+10) ≈ 0.1005 < ch0 0.30 -> aggregate improves
    agg = summ.aggregate_sessions(sessions)
    g = summ.evaluate_summary_gates(agg, sessions, all_evidence_valid=True)
    assert g["gates"]["aggregate_all8_cpcer_lt_ch0"] is True  # pooled still better
    assert g["gates"]["all_session_result_gates_passed"] is False  # BAD failed
    assert g["passed"] is False  # the batch must end RED


def test_low_recall_session_not_quality_fail_and_aggregate_passes():
    """Session-level DER/recall never mark a session quality_fail (aggregate only)."""
    sessions = []
    # 7 sessions with der 0.30 (>0.25) individually but only 2 frames each
    for i in range(7):
        s = gen.build_scores(
            all8=gen._variant((100, 1000), (0, 2000), (0, 9000)),
            ch0=gen._variant((300, 1000), (0, 2000), (0, 9000)),
            dia=gen._diarization(3, 10, 5, 10, None, None))  # der 0.30
        sessions.append(_session_dict(s, cid=f"D{i}"))
    agg = summ.aggregate_sessions(sessions)
    g = summ.evaluate_summary_gates(agg, sessions, all_evidence_valid=True)
    # aggregate DER = 21/70 = 0.30 > 0.25 -> aggregate gate FAILS
    assert g["gates"]["aggregate_der_le_25"] is False
    # but no session is quality_fail; the session-result gate stays green
    assert g["gates"]["all_session_result_gates_passed"] is True
    # flip: pool to a passing aggregate via one heavy clean session
    s_clean = gen.build_scores(
        all8=gen._variant((100, 1000), (0, 2000), (0, 9000)),
        ch0=gen._variant((300, 1000), (0, 2000), (0, 9000)),
        dia=gen._diarization(200, 10000, 5000, 6000, None, None))  # der 0.02
    sessions2 = sessions + [_session_dict(s_clean, cid="CLEAN")]
    agg2 = summ.aggregate_sessions(sessions2)
    g2 = summ.evaluate_summary_gates(agg2, sessions2, all_evidence_valid=True)
    assert g2["gates"]["aggregate_der_le_25"] is True


# ---------------------------------------------------------------------------
# 3. strict schema validation
# ---------------------------------------------------------------------------
def test_scores_case_id_mismatch_is_evidence_fail():
    scores, timing = _load("session_pass")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id="A_DIFFERENT_CASE")
    assert st["evidence_valid"] is False
    assert any("case_id" in r for r in st["failure_reasons"])


def test_gate_scope_not_full_is_evidence_fail():
    scores, timing = _load("session_pass")
    scores["gate_scope"] = "smoke"
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id=scores["case_id"])
    assert st["evidence_valid"] is False
    assert any("gate_scope" in r for r in st["failure_reasons"])
    # gates.gate_scope too
    scores2, t2 = _load("session_pass")
    scores2["gates"]["gate_scope"] = "auto"
    st2 = summ.classify_attempt(scores2, t2, verifier_passed=True, expected_hashes=None,
                                case_id=scores2["case_id"])
    assert st2["evidence_valid"] is False


def test_timing_missing_is_evidence_fail(tmp_path):
    """No full-loop-timing.json => outer RTF evidence absent => evidence failure."""
    root = tmp_path / "batch"
    d = root / "CASE0" / "attempt0"
    d.mkdir(parents=True)
    (d / "scores.json").write_text(json.dumps(_patch_case_id(_load("session_pass")[0], "CASE0")), encoding="utf-8")
    manifest = {"plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
                "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0", "verifier_passed": True}],
                                       "selected_attempt_dir": "CASE0/attempt0"}}}
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert sessions == []
    assert per_case["CASE0"]["status"]["evidence_valid"] is False
    assert any("timing" in r for r in per_case["CASE0"]["status"]["failure_reasons"])


def test_timing_wrong_type_is_evidence_fail(tmp_path):
    root = tmp_path / "batch"
    _copy_fixture(root, "CASE0", "session_pass")
    # corrupt the timing file type
    (root / "CASE0" / "attempt0" / "full-loop-timing.json").write_text(json.dumps(
        {"full_process_rtf_passed": "yes", "full_process_rtf": -1, "full_process_seconds": "x"}), encoding="utf-8")
    manifest = {"plan_id": "x", "cases": [{"order": 0, "case_id": "CASE0"}],
                "attempts": {"CASE0": {"attempts": [{"attempt_dir": "CASE0/attempt0", "verifier_passed": True}],
                                       "selected_attempt_dir": "CASE0/attempt0"}}}
    sessions, per_case, errors = summ.collect_selected(manifest, root, None)
    assert sessions == []
    assert any("full_process_rtf_passed" in r or "full_process_rtf" in r or "full_process_seconds" in r
               for r in per_case["CASE0"]["status"]["failure_reasons"])


def test_negative_cer_is_evidence_fail():
    scores, timing = _load("session_pass")
    scores["variants"]["all8"]["cpcer"]["cer"] = -0.1
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id=scores["case_id"])
    assert st["evidence_valid"] is False


def test_rate_inconsistent_with_numerator_is_evidence_fail():
    scores, timing = _load("session_pass")
    scores["variants"]["all8"]["cpcer"]["errors"] = 999  # now != cer * reference
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id=scores["case_id"])
    assert st["evidence_valid"] is False
    assert any("cpcer" in r for r in st["failure_reasons"])


# ---------------------------------------------------------------------------
# 4. privacy / replay evidence
# ---------------------------------------------------------------------------
def test_replay_evidence_green_on_pass_fixture():
    scores, _ = _load("session_pass")
    ok, reasons = summ.verify_replay_evidence(scores)
    assert ok, reasons


def test_replay_evidence_audit_embedding_fails():
    scores, _ = _load("session_privacy_embedding")
    ok, reasons = summ.verify_replay_evidence(scores)
    assert not ok
    assert any("embedding" in r for r in reasons)


def test_replay_evidence_not_interrupted_fails():
    scores, _ = _load("session_privacy_not_interrupted")
    ok, reasons = summ.verify_replay_evidence(scores)
    assert not ok
    assert any("interrupted" in r for r in reasons)


def test_replay_evidence_duplicate_ids_fail():
    replay = gen.build_replay(dedupe_ids=True)
    scores, timing = _load("session_pass")
    scores["service_replay"] = replay
    ok, reasons = summ.verify_replay_evidence(scores)
    assert not ok
    assert any("unique" in r for r in reasons)


def test_replay_evidence_count_mismatch_fails():
    replay = gen.build_replay(finals=8, chunks=7)  # chunk count disagrees
    scores, timing = _load("session_pass")
    scores["service_replay"] = replay
    ok, reasons = summ.verify_replay_evidence(scores)
    assert not ok
    assert any("capture_chunk_count" in r for r in reasons)


def test_privacy_leak_scores_are_evidence_fail():
    """A session whose replay evidence leaks is an evidence failure (aborts)."""
    scores, timing = _load("session_privacy_embedding")
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id=scores["case_id"])
    assert st["evidence_valid"] is False
    assert any("embedding" in r for r in st["failure_reasons"])


def test_driver_aborts_on_privacy_leak(tmp_path):
    """Batch driver: a privacy leak on session 1 is an evidence fail -> abort."""
    def leak_runner(cmd, timeout=None):
        if _invokes(cmd, "run_full_loop_timed.py"):
            _materialize(cmd, "session_privacy_embedding")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)

    rc = driver.main(_cli_args(tmp_path), runner=leak_runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 2
    batch_root = tmp_path / "out"
    manifest = json.loads((batch_root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["attempts"]) == 1  # stopped at the first session
    assert not (batch_root / "full8-summary.json").exists()


# ---------------------------------------------------------------------------
# 5. manifest lifecycle: batch-root refusal / drift / post_run
# ---------------------------------------------------------------------------
def test_existing_batch_root_is_refused_before_any_write(tmp_path):
    """A non-dry-run with an already-existing batch root refuses BEFORE writing."""
    batch = tmp_path / "out"
    batch.mkdir()
    (batch / "pre-existing.txt").write_text("x")
    with pytest.raises(SystemExit) as exc:
        driver.main(_cli_args(tmp_path, extra=[]), runner=FakeRunner(case_id=None),
                    git=FakeGit(), env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert "already exists" in str(exc.value)
    assert (batch / "pre-existing.txt").read_text() == "x"
    assert not (batch / "batch-run-manifest.json").exists()


def test_insufficient_disk_is_preflight_failure(tmp_path):
    """The disk gate is a REAL-batch preflight gate (dry-run writes nothing and
    only reports availability)."""
    # batch root must not exist for the real run to reach preflight
    with pytest.raises(SystemExit) as exc:
        driver.main(_cli_args(tmp_path, extra=[]), runner=FakeRunner(case_id=None),
                    git=FakeGit(), env_probe=FakeEnvProbe(), free_disk_gib=1.0)
    assert "insufficient disk space" in str(exc.value)
    # dry-run reports but never hard-fails on low disk
    rc = driver.main(_cli_args(tmp_path, extra=["--dry-run"]), runner=None, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=1.0)
    assert rc == 0


class DriftGit(FakeGit):
    """FakeGit that reports a HEAD move AFTER the batch baseline snapshot.

    head_sha is read several times before the loop (preflight, prefs, the
    snapshot_before baseline), so the drift only starts on the 4th read — i.e.
    right at the first post-session re-check.
    """

    def __init__(self):
        self.calls = 0

    def head_sha(self):
        self.calls += 1
        return "abc123def456" if self.calls <= 3 else "deadbeefcafe"


def test_head_drift_stops_batch_without_green_summary(tmp_path):
    """A HEAD change between sessions stops the batch; no summary is produced."""
    runner = FakeRunner(case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=DriftGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 2
    batch_root = tmp_path / "out"
    manifest = json.loads((batch_root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["attempts"]) == 1  # stopped right after the first session
    assert manifest["post_run"]["status"] == "aborted"
    assert manifest["post_run"]["code_drift"]  # non-empty drift recorded
    assert not (batch_root / "full8-summary.json").exists()


def test_post_run_recorded_on_green_batch(tmp_path):
    runner = FakeRunner(case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    batch_root = tmp_path / "out"
    manifest = json.loads((batch_root / "batch-run-manifest.json").read_text(encoding="utf-8"))
    pr = manifest["post_run"]
    assert pr["finished_at"] and pr["head_sha"] == "abc123def456"
    assert pr["status"] == "completed" and pr["code_drift"] == []
    assert pr["runtime_sources_locked"] is True
    # the three input manifests are hash-locked in the ledger
    locks = manifest["locks"]
    assert locks["prep_manifest_sha256"]
    assert locks["prediction_run_manifest_sha256"]
    assert locks["source_run_manifest_sha256"]


# ===========================================================================
# Round-3 additions: six hardening items — reverse tests (each bad input must
# produce a NON-ZERO failure; never "passed")
# ===========================================================================

# A realistic (illustrative) runtime source hash set: the SAME set, with identical
# per-file values, recorded both before (manifest.git) and after (post_run) so the
# summarizer's before/after comparison stays green. Missing/empty/mismatched is what
# the reverse tests below exercise.
SRC_HASH_SET = {
    "evals/audio_frontend/run_continuous_full8.py": "a" * 64,
    "evals/audio_frontend/summarize_continuous_full8.py": "b" * 64,
    "evals/audio_frontend/run_full_loop_timed.py": "c" * 64,
    "evals/audio_frontend/verify_frontend_wavs.py": "d" * 64,
    "evals/audio_frontend/run_continuous_frontend.py": "e" * 64,
    "evals/audio_frontend/score_continuous_e2e.py": "f" * 64,
    "evals/audio_frontend/score_diarization.py": "11" * 32,
    "evals/audio_frontend/diarize_sortformer.py": "22" * 32,
    "evals/audio_frontend/enhance_mvdr_oracle.py": "33" * 32,
    "ai_glasses_memory_assistant/audio_engine/backends.py": "44" * 32,
    "ai_glasses_memory_assistant/evals/eval_ali.py": "55" * 32,
    "ai_glasses_memory_assistant/agent_bridge.py": "66" * 32,
    "scripts/p0_oracle_interval_ablation.py": "77" * 32,
}


def _build_valid_manifest(tmp_path):
    """A fully valid 8-session manifest with realized products on disk."""
    root = tmp_path / "batch"
    for cid in gen.CASES:
        _copy_fixture(root, cid, "session_pass")
    attempts = {}
    for cid in gen.CASES:
        attempts[cid] = {
            "attempts": [{"attempt": 0, "attempt_dir": f"{cid}/attempt0",
                          "verifier_passed": True}],
            "selected_attempt_dir": f"{cid}/attempt0",
        }
    manifest = {
        "schema": summ.SCHEMA,
        "plan_id": summ.PLAN_ID,
        "cases": [{"order": i, "case_id": cid} for i, cid in enumerate(gen.CASES)],
        "attempts": attempts,
        # git block carries the REAL required fields the summarizer now enforces:
        # an explicitly empty status and a strictly True runtime_sources_match_head.
        "git": {"head_sha": "H" * 40, "status": "",
                "runtime_sources_match_head": True,
                "runtime_source_sha256": dict(SRC_HASH_SET)},
        "locks": {"model_hashes": {
            "asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
            "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256,
        }},
        "post_run": {
            "status": "completed", "finished_at": "2026-09-08T00:00:00Z",
            "head_sha": "H" * 40, "code_drift": [], "git_status": "",
            "runtime_sources_locked": True,
            "runtime_source_sha256": dict(SRC_HASH_SET),
        },
    }
    (root / "batch-run-manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def _integrity_manifest(tmp_path, **overrides):
    root, manifest = _build_valid_manifest(tmp_path)
    for k, v in overrides.items():
        manifest[k] = v
    return root, manifest


# ---------------------------------------------------------------------------
# 1. strict manifest integrity: one / seven / duplicate / aborted / drift /
#    missing model lock / head-moved / non-empty git status => non-zero failure
# ---------------------------------------------------------------------------
def test_manifest_integrity_clean_base(tmp_path):
    root, manifest = _build_valid_manifest(tmp_path)
    assert summ.check_manifest_integrity(manifest) == []


def test_manifest_one_session_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["cases"] = [manifest["cases"][0]]
    assert summ.check_manifest_integrity(manifest)


def test_manifest_seven_sessions_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["cases"] = manifest["cases"][:7]
    assert summ.check_manifest_integrity(manifest)


def test_manifest_duplicate_case_id_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["cases"][3]["case_id"] = manifest["cases"][0]["case_id"]
    assert summ.check_manifest_integrity(manifest)


def test_manifest_post_run_aborted_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["post_run"]["status"] = "aborted"
    assert summ.check_manifest_integrity(manifest)


def test_manifest_code_drift_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["post_run"]["code_drift"] = ["runtime sources changed mid-batch"]
    assert summ.check_manifest_integrity(manifest)


def test_manifest_missing_model_lock_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["locks"]["model_hashes"] = {}
    assert summ.check_manifest_integrity(manifest)


def test_manifest_model_lock_wrong_value_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["locks"]["model_hashes"]["asr_model_sha256"] = "DEADBEEF"
    assert summ.check_manifest_integrity(manifest)


def test_manifest_head_moved_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["post_run"]["head_sha"] = "F" * 40
    assert summ.check_manifest_integrity(manifest)


def test_manifest_nonempty_git_status_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    manifest["post_run"]["git_status"] = " M evals/audio_frontend/run_continuous_full8.py"
    assert summ.check_manifest_integrity(manifest)


# --- schema / plan_id must match EXACTLY (wrong or missing => fail-closed) ------
def test_manifest_schema_wrong_value_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in ("continuous_full8_batch.v2", "continuous_full8_batch.v3 ",
                "CONTINUOUS_FULL8_BATCH.V3", ""):
        manifest["schema"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("schema" in p for p in problems), bad


def test_manifest_schema_missing_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["schema"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("schema" in p for p in problems)


def test_manifest_schema_wrong_type_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (None, 3, ["continuous_full8_batch.v3"], {"v": 3}):
        manifest["schema"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("schema" in p for p in problems), bad


def test_manifest_plan_id_wrong_value_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in ("2026-09-08-audio-continuous-e2e-full7",
                "2026-09-08-audio-continuous-e2e-full8 ", ""):
        manifest["plan_id"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("plan_id" in p for p in problems), bad


def test_manifest_plan_id_missing_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["plan_id"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("plan_id" in p for p in problems)


def test_manifest_plan_id_wrong_type_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (None, 8, ["2026-09-08-audio-continuous-e2e-full8"]):
        manifest["plan_id"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("plan_id" in p for p in problems), bad


def test_schema_and_plan_id_have_single_authoritative_definition():
    """Driver re-exports the summarizer's constants: writer and validator cannot drift."""
    assert driver.SCHEMA is summ.SCHEMA == "continuous_full8_batch.v3"
    assert driver.PLAN_ID is summ.PLAN_ID == "2026-09-08-audio-continuous-e2e-full8"
    assert summ.LOCKED_ASR_MODEL_SHA256 == driver.LOCKED_ASR_MODEL_SHA256
    assert summ.LOCKED_SORTFORMER_MODEL_SHA256 == driver.LOCKED_SORTFORMER_MODEL_SHA256


# --- git.status must be explicitly empty; match_head strictly True --------------
def test_manifest_git_status_missing_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["git"]["status"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("git.status" in p for p in problems)


def test_manifest_git_status_wrong_type_or_nonempty_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (None, [], 0, " M notes.md"):
        manifest["git"]["status"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("git.status" in p for p in problems), bad


def test_manifest_runtime_sources_match_head_missing_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["git"]["runtime_sources_match_head"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("runtime_sources_match_head" in p for p in problems)


def test_manifest_runtime_sources_match_head_false_or_wrong_type_fails(tmp_path):
    """Truthy-but-not-True values (1, "yes") must also fail: strict ``is True``."""
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (False, None, 1, "yes", "true"):
        manifest["git"]["runtime_sources_match_head"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("runtime_sources_match_head" in p for p in problems), bad


# --- real run refuses before start when a NON-Python file dirties the tree ------
class DirtyNonPythonGit(FakeGit):
    """All managed Python sources are HEAD-clean, but a non-Python file changed."""

    def status_porcelain(self):
        return " M notes.md\n?? data/scratch.bin"


def test_real_run_refuses_when_non_python_file_dirties_tree(tmp_path):
    """A modified/untracked NON-Python file must refuse before eight sessions run."""
    repo_root = AUDIO_FRONTEND.parent.parent
    git = DirtyNonPythonGit()
    with pytest.raises(SystemExit) as exc:
        driver.preflight(
            repo_root, tmp_path / "out", git, enforce_git=True,
            expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                             "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
            free_disk_gib=_BIG_DISK, max_retries=2,
            total_processed_seconds=15138.215)
    assert "not clean" in str(exc.value)
    assert "notes.md" in str(exc.value)


def test_dry_run_tolerates_dirty_tree_but_records_it(tmp_path):
    """Dry-run still previews (it writes nothing) but the status is recorded."""
    repo_root = AUDIO_FRONTEND.parent.parent
    pf = driver.preflight(
        repo_root, tmp_path / "out", DirtyNonPythonGit(), enforce_git=False,
        expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                         "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
        free_disk_gib=_BIG_DISK, max_retries=2,
        total_processed_seconds=15138.215)
    assert pf["status"] == " M notes.md\n?? data/scratch.bin"


# --- strict post_run typing + explicit-empty semantics (missing != clean) ------
def test_manifest_code_drift_missing_fails(tmp_path):
    """A MISSING code_drift is not 'clean' — it means no verdict was recorded."""
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["post_run"]["code_drift"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("code_drift" in p for p in problems)


def test_manifest_code_drift_wrong_type_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in ("", False, 0, None, {"drift": False}):
        manifest["post_run"]["code_drift"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("code_drift" in p for p in problems), bad


def test_manifest_git_status_missing_fails(tmp_path):
    """A MISSING git_status cannot be assumed clean."""
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["post_run"]["git_status"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("git_status" in p for p in problems)


def test_manifest_git_status_wrong_type_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in ([], False, 0, None, {"clean": True}):
        manifest["post_run"]["git_status"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("git_status" in p for p in problems), bad


def test_manifest_finished_at_missing_or_wrong_type_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (None, "", 12345, True):
        manifest["post_run"]["finished_at"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("finished_at" in p for p in problems), bad


def test_manifest_runtime_sources_locked_not_bool_true_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in (None, False, 1, "yes"):
        manifest["post_run"]["runtime_sources_locked"] = bad
        assert summ.check_manifest_integrity(manifest), bad


# --- before/after source hash set: complete, same set, identical values --------
def test_manifest_source_hash_missing_after_set_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["post_run"]["runtime_source_sha256"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("hash set incomplete" in p for p in problems)


def test_manifest_source_hash_missing_before_set_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    del manifest["git"]["runtime_source_sha256"]
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("hash set incomplete" in p for p in problems)


def test_manifest_source_hash_missing_file_fails(tmp_path):
    """A file present before but absent after = mid-batch source churn."""
    root, manifest = _integrity_manifest(tmp_path)
    manifest["post_run"]["runtime_source_sha256"].pop(
        "evals/audio_frontend/enhance_mvdr_oracle.py")
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("hash set differs" in p for p in problems)
    assert any("enhance_mvdr_oracle.py" in p for p in problems)


def test_manifest_source_hash_value_mismatch_fails(tmp_path):
    """Same file set, but one hash changed = undetected code drift."""
    root, manifest = _integrity_manifest(tmp_path)
    rel = "ai_glasses_memory_assistant/agent_bridge.py"
    manifest["post_run"]["runtime_source_sha256"][rel] = "f" * 64
    problems = summ.check_manifest_integrity(manifest)
    assert problems and any("hash mismatch" in p and rel in p for p in problems)


def test_manifest_source_hash_empty_value_fails(tmp_path):
    """A present key with an empty/None hash is a missing hash, not a match."""
    root, manifest = _integrity_manifest(tmp_path)
    rel = "evals/audio_frontend/run_continuous_full8.py"
    for bad in ("", None):
        manifest["post_run"]["runtime_source_sha256"][rel] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("hash mismatch" in p and rel in p for p in problems), bad


def test_manifest_source_hash_set_malformed_fails(tmp_path):
    root, manifest = _integrity_manifest(tmp_path)
    for bad in ([], "not-a-dict", 42):
        manifest["post_run"]["runtime_source_sha256"] = bad
        problems = summ.check_manifest_integrity(manifest)
        assert problems and any("malformed" in p or "incomplete" in p for p in problems), bad


def test_main_returns_nonzero_on_one_session_manifest(tmp_path):
    """End-to-end: a 1-session manifest must yield a non-zero summary exit."""
    root, manifest = _integrity_manifest(tmp_path)
    manifest["cases"] = [manifest["cases"][0]]
    keep = manifest["cases"][0]["case_id"]
    manifest["attempts"] = {keep: manifest["attempts"][keep]}
    mpath = root / "batch-run-manifest.json"
    mpath.write_text(json.dumps(manifest))
    rc = summ.main(["--manifest", str(mpath), "--out", str(tmp_path / "summary.json")])
    assert rc == 1


# ---------------------------------------------------------------------------
# 2. RTF authenticity: a falsified passed flag (rtf=9 but passed=True) is rejected
# ---------------------------------------------------------------------------
def test_rtf_authenticity_rejects_falsified_outer_passed():
    scores, timing = _load("session_pass")
    timing["full_process_seconds"] = 1.0
    timing["processed_seconds"] = 0.111            # outer rtf recomputes to ~9.0
    timing["full_process_rtf"] = 9.0
    timing["full_process_rtf_passed"] = True       # FALSE claim: real rtf > 1.0
    problems = summ.check_rtf_authenticity(scores, timing)
    assert problems, "should flag the falsified passed flag"
    assert any("passed flag" in p for p in problems)
    st = summ.classify_attempt(scores, timing, verifier_passed=True, expected_hashes=None,
                               case_id=scores["case_id"])
    assert st["quality_passed"] is False
    assert any("rtf" in r for r in st["failure_reasons"])


def test_rtf_authenticity_rejects_falsified_inner_passed():
    scores, timing = _load("session_pass")
    scores["full_loop_rtf"] = 9.0                  # real inner sum is 0.35
    scores["gates"]["checks"]["rtf"] = True        # FALSE claim
    problems = summ.check_rtf_authenticity(scores, timing)
    assert any("inner rtf" in p for p in problems)


def test_rtf_authenticity_clean_on_valid_fixture():
    scores, timing = _load("session_pass")
    assert summ.check_rtf_authenticity(scores, timing) == []


# ---------------------------------------------------------------------------
# 3. source dependency lock: entry unchanged but an imported dependency (ASR
#    backend / CER) changed must refuse to start
# ---------------------------------------------------------------------------
class DepDriftGit(driver.GitInspector):
    """matches_head fails ONLY for the ASR backend + oracle-ablation dependencies."""
    _BAD = (
        "ai_glasses_memory_assistant/audio_engine/backends.py",
        "scripts/p0_oracle_interval_ablation.py",
    )

    def is_tracked(self, rel):
        return True

    def matches_head(self, rel):
        # preflight passes absolute paths, so compare on the relative tail
        return not any(str(rel).endswith(b) for b in self._BAD)


def test_dependency_source_drift_refuses_preflight(tmp_path):
    repo_root = AUDIO_FRONTEND.parent.parent
    git = DepDriftGit(repo_root=repo_root)
    with pytest.raises(SystemExit):
        driver.preflight(
            repo_root, tmp_path / "out", git, enforce_git=True,
            expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                             "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
            free_disk_gib=_BIG_DISK, max_retries=2)


def test_dependency_sources_recorded_in_manifest(tmp_path):
    runner = FakeRunner(case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "batch-run-manifest.json").read_text())
    # The source lock now covers ALL managed *.py under the three locked roots
    # (directory enumeration), so the recorded set is large and includes the
    # specifically import-dependent files plus MVDR / agent_bridge.
    src = manifest["git"]["runtime_source_sha256"]
    assert len(src) >= 80
    for rel in ("evals/audio_frontend/score_diarization.py",
                "evals/audio_frontend/diarize_sortformer.py",
                "evals/audio_frontend/enhance_mvdr_oracle.py",
                "ai_glasses_memory_assistant/audio_engine/backends.py",
                "ai_glasses_memory_assistant/evals/eval_ali.py",
                "ai_glasses_memory_assistant/agent_bridge.py",
                "scripts/p0_oracle_interval_ablation.py"):
        assert rel in src, rel
    assert manifest["post_run"]["runtime_sources_locked"] is True


# --- drift in ANY managed run-code file must be caught, not just a hand-picked
#     import list: the MVDR enhancer and agent_bridge are the two named cases ----
class SingleFileDriftGit(driver.GitInspector):
    """matches_head fails for exactly one managed source file (given by tail)."""

    def __init__(self, bad_tail, repo_root=None):
        super().__init__(repo_root=repo_root)
        self.bad_tail = bad_tail

    def head_sha(self):
        return "H" * 40

    def status_porcelain(self):
        return ""

    def is_tracked(self, rel):
        return True

    def matches_head(self, rel):
        # preflight passes absolute paths, so compare on the relative tail
        return not str(rel).endswith(self.bad_tail)


def test_mvdr_implementation_drift_refuses_preflight(tmp_path):
    """A changed MVDR enhancer implementation must refuse the run before it starts."""
    repo_root = AUDIO_FRONTEND.parent.parent
    git = SingleFileDriftGit("evals/audio_frontend/enhance_mvdr_oracle.py",
                             repo_root=repo_root)
    with pytest.raises(SystemExit) as exc:
        driver.preflight(
            repo_root, tmp_path / "out", git, enforce_git=True,
            expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                             "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
            free_disk_gib=_BIG_DISK, max_retries=2,
            total_processed_seconds=8 * 26 * 60)
    assert "runtime sources" in str(exc.value) or "git lock failed" in str(exc.value)


def test_agent_bridge_drift_refuses_preflight(tmp_path):
    """A changed agent_bridge must refuse the run before it starts."""
    repo_root = AUDIO_FRONTEND.parent.parent
    git = SingleFileDriftGit("ai_glasses_memory_assistant/agent_bridge.py",
                             repo_root=repo_root)
    with pytest.raises(SystemExit) as exc:
        driver.preflight(
            repo_root, tmp_path / "out", git, enforce_git=True,
            expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                             "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
            free_disk_gib=_BIG_DISK, max_retries=2,
            total_processed_seconds=8 * 26 * 60)
    assert "runtime sources" in str(exc.value) or "git lock failed" in str(exc.value)


class DriftFlag:
    def __init__(self):
        self.armed = False


class MidBatchDriftGit(driver.GitInspector):
    """Clean until ``flag.armed`` is set, then reports the given file as drifted."""

    def __init__(self, flag, bad_tail, repo_root=None):
        super().__init__(repo_root=repo_root)
        self.flag = flag
        self.bad_tail = bad_tail

    def head_sha(self):
        return "H" * 40

    def status_porcelain(self):
        return f" M {self.bad_tail}" if self.flag.armed else ""

    def is_tracked(self, rel):
        return True

    def matches_head(self, rel):
        return not (self.flag.armed and str(rel).endswith(self.bad_tail))


class ArmDriftAfterFirstSession(FakeRunner):
    """Arms the drift flag once the first session's wrapper has been launched."""

    def __init__(self, flag, **kw):
        super().__init__(**kw)
        self.flag = flag

    def __call__(self, cmd, timeout=None):
        out = super().__call__(cmd, timeout=timeout)
        if _invokes(cmd, "run_full_loop_timed.py"):
            self.flag.armed = True
        return out


def test_midbatch_mvdr_drift_stops_before_next_session(tmp_path):
    """Drift appearing after session 1 stops the batch; session 2 never starts."""
    flag = DriftFlag()
    git = MidBatchDriftGit(flag, "evals/audio_frontend/enhance_mvdr_oracle.py",
                           repo_root=AUDIO_FRONTEND.parent.parent)
    runner = ArmDriftAfterFirstSession(flag, case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=git,
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc != 0
    wrapper_calls = [c for c, _ in runner.calls if _invokes(c, "run_full_loop_timed.py")]
    assert len(wrapper_calls) == 1, f"session 2 must not start, got {len(wrapper_calls)}"
    manifest = json.loads((tmp_path / "out" / "batch-run-manifest.json").read_text())
    assert manifest["post_run"]["status"] == "aborted"
    assert manifest["post_run"]["code_drift"]
    # the summarizer must also reject this manifest (no green summary from drift)
    assert summ.check_manifest_integrity(manifest)


def test_midbatch_agent_bridge_drift_stops_before_next_session(tmp_path):
    flag = DriftFlag()
    git = MidBatchDriftGit(flag, "ai_glasses_memory_assistant/agent_bridge.py",
                           repo_root=AUDIO_FRONTEND.parent.parent)
    runner = ArmDriftAfterFirstSession(flag, case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=git,
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc != 0
    wrapper_calls = [c for c, _ in runner.calls if _invokes(c, "run_full_loop_timed.py")]
    assert len(wrapper_calls) == 1, f"session 2 must not start, got {len(wrapper_calls)}"


# ---------------------------------------------------------------------------
# 4. timeout kills the WHOLE subprocess group; max-retries clamped to 0..2;
#    interrupt never retries
# ---------------------------------------------------------------------------
def test_max_retries_clamped_end_to_end(tmp_path):
    # --max-retries 9 must be clamped to 2 => attempt0..attempt2 (3 launches) then abort
    runner = FakeRunner(raise_on_wrapper_call=99)
    rc = driver.main(_cli_args(tmp_path, extra=["--max-retries", "9"]),
                     runner=runner, git=FakeGit(), env_probe=FakeEnvProbe(),
                     free_disk_gib=_BIG_DISK)
    assert rc == 2
    wrapper_calls = [c for c, _ in runner.calls if _invokes(c, "run_full_loop_timed.py")]
    assert len(wrapper_calls) == 3


def test_max_retries_zero_allows_no_retry(tmp_path):
    runner = FakeRunner(raise_on_wrapper_call=99)
    rc = driver.main(_cli_args(tmp_path, extra=["--max-retries", "0"]),
                     runner=runner, git=FakeGit(), env_probe=FakeEnvProbe(),
                     free_disk_gib=_BIG_DISK)
    assert rc == 2
    wrapper_calls = [c for c, _ in runner.calls if _invokes(c, "run_full_loop_timed.py")]
    assert len(wrapper_calls) == 1


def test_subprocess_group_killed_no_orphans(tmp_path):
    """A timeout/interrupt must kill the whole process group, leaving no orphaned
    descendant. Verified with a no-audio parent shell that backgrounds a sleep."""
    pidfile = tmp_path / "gc.pid"
    child = subprocess.Popen(
        ["sh", "-c", f"( sleep 60 & echo $! > {pidfile} ); wait"],
        start_new_session=True,
    )
    pgid = child.pid
    gc_pid = None
    for _ in range(200):
        if pidfile.exists():
            txt = pidfile.read_text().strip()
            if txt:
                gc_pid = int(txt)
                break
        time.sleep(0.05)
    assert gc_pid is not None, "grandchild pid never written"
    # simulate the driver's kill-on-timeout: kill the ENTIRE process group
    driver._kill_process_group(pgid, signal.SIGTERM)
    child.wait(timeout=10)
    # the backgrounded grandchild must be gone too (no orphaned descendant survives)
    deadline = time.time() + 5
    gone = False
    while time.time() < deadline:
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.05)
    assert gone, "orphaned grandchild still alive after group kill"


# ---------------------------------------------------------------------------
# 5. disk budget derived from output structure (retries x sessions + margin);
#    unreadable space must NOT be marked budget_ok; derivation written to manifest
# ---------------------------------------------------------------------------
def test_disk_budget_derived_from_attempts_and_sessions(tmp_path):
    """Budget is DERIVED from real output structure, not a magic per-session GiB."""
    repo_root = AUDIO_FRONTEND.parent.parent
    # 8 sessions x 26 min processed audio (the real per-session order of magnitude).
    total_s = 8 * 26 * 60
    pf = driver.preflight(
        repo_root, tmp_path / "out", FakeGit(), enforce_git=False,
        expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                         "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
        free_disk_gib=_BIG_DISK, max_retries=2,
        total_processed_seconds=total_s)
    basis = pf["disk_budget_basis"]
    # every derived term is recorded, not just the final number
    assert basis["total_processed_seconds"] == float(total_s)
    assert basis["sample_rate"] == driver.WAV_SAMPLE_RATE
    assert basis["bytes_per_sample"] == driver.WAV_BYTES_PER_SAMPLE
    assert basis["max_tracks"] == driver.WAV_MAX_TRACKS
    assert basis["num_variants"] == driver.WAV_NUM_VARIANTS
    assert basis["retry_factor"] == 3                 # max_retries 2 + 1 (retry retention)
    assert basis["n_sessions"] == 8
    # WAV upper bound: mono PCM_16 per track — seconds x rate x bytes x 4 TRACKS x
    # 2 variants, converted with 1024**3 (GiB), never the decimal 1e9 GB.
    expected_wav_gib = (total_s * driver.WAV_SAMPLE_RATE * driver.WAV_BYTES_PER_SAMPLE
                        * driver.WAV_MAX_TRACKS * driver.WAV_NUM_VARIANTS) / driver.GIB
    assert basis["max_tracks"] == 4
    assert basis["wav_gib"] == round(expected_wav_gib, 6)
    expected_text_log = driver.TEXT_LOG_MARGIN_GIB_PER_SESSION * 8
    assert basis["text_log_gib"] == round(expected_text_log, 6)
    expected_total = ((expected_wav_gib + expected_text_log) * 3
                      + driver.DISK_SAFETY_MARGIN_GIB)
    assert basis["total_needed_gib"] == round(expected_total, 6)
    # the old unjustified default is gone for good
    assert not hasattr(driver, "DISK_BUDGET_GB_PER_SESSION")
    assert basis["derivation"]
    # derivation must NOT double-count: the 8 sessions are already inside wav_gib and
    # text_log_gib, so only the retry factor multiplies on top of them.
    assert "all 8 sessions" in basis["derivation"]
    assert "x 3 attempts" in basis["derivation"]
    assert "x 3 attempts x 8 sessions" not in basis["derivation"]
    assert pf["disk_budget_ok"] is True


def test_wav_model_is_four_tracks_two_variants():
    """Mono PCM_16 per anonymous track, Sortformer caps at 4 tracks, 2 variants."""
    assert driver.WAV_MAX_TRACKS == 4
    assert driver.WAV_NUM_VARIANTS == 2
    assert driver.WAV_BYTES_PER_SAMPLE == 2      # PCM_16
    assert driver.WAV_SAMPLE_RATE == 16_000
    # 4 tracks x 2 variants, NOT an 8-channel file
    assert not hasattr(driver, "WAV_MAX_CHANNELS")


def test_disk_budget_uses_gib_not_decimal_gb():
    """All disk figures are GiB (1024**3); 1 GiB of audio must not read as 1.074 GB."""
    assert driver.GIB == 1024 ** 3
    b = driver.derive_disk_budget(1024 ** 3 / (16000 * 2 * 4 * 2))  # exactly 1 GiB of WAV
    assert b["wav_gib"] == 1.0, b["wav_gib"]
    # decimal-GB conversion would understate the same bytes by ~7.4%
    assert abs(b["wav_gib"] * driver.GIB / 1e9 - 1.073741824) < 1e-6


def test_disk_budget_real_prep_manifest_retry2_about_20_gib(tmp_path):
    """Real eight-session duration 15138.215 s, default max_retries=2 -> ~20.028 GiB."""
    repo_root = AUDIO_FRONTEND.parent.parent
    total_s = 15138.215
    pf = driver.preflight(
        repo_root, tmp_path / "out", FakeGit(), enforce_git=False,
        expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                         "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
        free_disk_gib=_BIG_DISK, max_retries=2,
        total_processed_seconds=total_s)
    basis = pf["disk_budget_basis"]
    assert basis["total_processed_seconds"] == 15138.215
    # WAV 3.609 GiB + logs 0.400 GiB = 4.009 GiB/attempt; x3 + 8 GiB = 20.028 GiB
    assert round(basis["wav_gib"], 3) == 3.609, basis["wav_gib"]
    assert round(basis["text_log_gib"], 3) == 0.400
    assert round(basis["per_attempt_total_gib"], 3) == 4.009, basis["per_attempt_total_gib"]
    assert round(basis["total_needed_gib"], 3) == 20.028, basis["total_needed_gib"]
    assert pf["disk_budget_ok"] is True


def test_disk_budget_real_run_refuses_when_params_invalid(tmp_path):
    """Real run (enforce_git=True) refuses when the budget cannot be derived."""
    repo_root = AUDIO_FRONTEND.parent.parent
    for bad in (None, 0, -1.0):
        with pytest.raises(SystemExit) as exc:
            driver.preflight(
                repo_root, tmp_path / "out", FakeGit(), enforce_git=True,
                expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                                 "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
                free_disk_gib=_BIG_DISK, max_retries=2,
                total_processed_seconds=bad)
        assert "disk budget cannot be derived" in str(exc.value)


def test_disk_budget_real_run_refuses_when_space_insufficient(tmp_path):
    """Real run refuses on insufficient space; it must NOT be bypassable by CLI."""
    repo_root = AUDIO_FRONTEND.parent.parent
    total_s = 8 * 26 * 60
    with pytest.raises(SystemExit) as exc:
        driver.preflight(
            repo_root, tmp_path / "out", FakeGit(), enforce_git=True,
            expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                             "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
            free_disk_gib=1.0, max_retries=2, total_processed_seconds=total_s)
    assert "insufficient disk space" in str(exc.value)
    # no CLI knob left to lower the budget and sneak past the gate
    import argparse
    src = AUDIO_FRONTEND.joinpath("run_continuous_full8.py").read_text(encoding="utf-8")
    assert "disk-budget-gb-per-session" not in src


def test_disk_budget_unreadable_not_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "disk_free_gib", lambda path: None)
    repo_root = AUDIO_FRONTEND.parent.parent
    pf = driver.preflight(
        repo_root, tmp_path / "out", FakeGit(), enforce_git=False,
        expected_hashes={"asr_model_sha256": driver.LOCKED_ASR_MODEL_SHA256,
                         "sortformer_model_sha256": driver.LOCKED_SORTFORMER_MODEL_SHA256},
        free_disk_gib=None, max_retries=2)
    assert pf["disk_free_gib"] is None
    assert pf["disk_budget_ok"] is False


def test_disk_budget_written_to_manifest(tmp_path):
    runner = FakeRunner(case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "batch-run-manifest.json").read_text())
    db = manifest["disk_budget"]
    assert db["budget_ok"] is True
    assert db["retry_factor"] == 3
    assert db["derivation"]
    assert db["available_gib"] == _BIG_DISK
    # every GiB term is spelled GiB (never a decimal GB mislabelled as GiB)
    for key in ("wav_gib", "text_log_gib", "per_attempt_total_gib",
                "safety_margin_gib", "available_gib", "needed_gib"):
        assert key in db, key
    assert db["max_tracks"] == 4
    assert db["needed_gib"] == db["per_attempt_total_gib"] * 3 + db["safety_margin_gib"]
    # existing products are NEVER deleted to free space: the summary still reflects a
    # full 8-session aggregate rather than a truncated one
    summary = json.loads((tmp_path / "out" / "full8-summary.json").read_text())
    assert summary["aggregate"]["variants"]["all8"]["cpcer"]["errors"] == 245 * 8


# ---------------------------------------------------------------------------
# 6. environment probe failure stops BEFORE the wrapper launches; all three used
#    env interpreters (frontend/scorer/wrapper) are recorded; resolved conda is
#    passed explicitly to the wrapper command
# ---------------------------------------------------------------------------
class FailingEnvProbe:
    """Resolves only the "hermes" env; the "py311" frontend env fails."""
    def __call__(self, conda, env):
        if env == "hermes":
            return "/usr/bin/python3", "3.11.0"
        return None, None


def test_env_probe_failure_stops_before_launch(tmp_path):
    class BoomRunner:
        def __call__(self, cmd, timeout=None):
            raise AssertionError("wrapper must not be launched when probe fails")

    with pytest.raises(SystemExit) as exc:
        driver.main(_cli_args(tmp_path), runner=BoomRunner(), git=FakeGit(),
                     env_probe=FailingEnvProbe(), free_disk_gib=_BIG_DISK)
    assert "environment probe failed" in str(exc.value)
    assert not (tmp_path / "out" / "batch-run-manifest.json").exists()


def test_all_three_env_interpreters_recorded(tmp_path):
    runner = FakeRunner(case_id=None)
    rc = driver.main(_cli_args(tmp_path), runner=runner, git=FakeGit(),
                     env_probe=FakeEnvProbe(), free_disk_gib=_BIG_DISK)
    assert rc == 0
    manifest = json.loads((tmp_path / "out" / "batch-run-manifest.json").read_text())
    for key in ("frontend_python", "scorer_python", "wrapper_python"):
        assert manifest["env"][key] == "/usr/bin/python3", key


def test_wrapper_command_uses_resolved_conda():
    cmd = driver.build_wrapper_command(
        conda="/resolved/conda", wrapper_env="hermes",
        repo_root=AUDIO_FRONTEND.parent.parent, prep_dir=FIX, predictions=FIX,
        source_run=FIX, case_id="R1", attempt_dir=Path("OUT"),
        frontend_env="py311", scorer_env="hermes", replay_timeout=60.0)
    assert cmd[0] == "/resolved/conda"     # resolved conda passed explicitly
    assert "-n" in cmd
