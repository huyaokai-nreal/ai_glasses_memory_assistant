from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ai_glasses_memory_assistant.evals.eval_ali import EvalAliCase, ReferenceInterval


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "p1_eval_ali_overlap_baseline.py"
    spec = importlib.util.spec_from_file_location("p1_eval_ali_overlap_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_aggregate_reports_oracle_delta_and_outcomes() -> None:
    module = _module()
    rows = [
        module.PairedIntervalResult("case", "N_SPK1", 0.0, 1.0, 1.0, True, 10, 6, 1, 5, 0, 1, 1, 0, 0),
        module.PairedIntervalResult("case", "N_SPK1", 2.0, 3.0, 1.0, False, 10, 2, 1, 1, 0, 2, 1, 1, 0),
    ]

    result = module.aggregate_paired(rows)

    assert result["far_mixed"]["normalized_cer"] == 0.4
    assert result["near_oracle"]["normalized_cer"] == 0.15
    assert result["near_minus_far"]["cer_absolute"] == -0.25
    assert result["near_minus_far"]["deletion_rate_absolute"] == -0.25
    assert result["paired_interval_outcomes"] == {"near_better": 1, "equal": 1, "near_worse": 0}


def test_pair_result_rejects_reference_mismatch() -> None:
    module = _module()
    far = module.IntervalResult("case", "N_SPK1", 0.0, 1.0, 1.0, True, 10, 2, 1, 1, 0)
    near = module.IntervalResult("case", "N_SPK1", 0.0, 1.0, 1.0, True, 9, 1, 1, 0, 0)

    try:
        module.pair_result(far, near)
    except ValueError as exc:
        assert "not aligned" in str(exc)
    else:
        raise AssertionError("reference mismatch must fail")


def test_near_gold_accepts_selected_far_window(tmp_path: Path) -> None:
    module = _module()
    selected = ReferenceInterval("N_SPK1", 1.0, 2.0, "窗口内")
    case = EvalAliCase("case", "session", "audio", "grid", "a", "b", 5.0, 0.0, 3.0, (selected,))
    textgrid = tmp_path / "near.TextGrid"
    textgrid.write_text(
        '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "c1"
        xmin = 0
        xmax = 5
        intervals: size = 2
        intervals [1]:
            xmin = 1
            xmax = 2
            text = "窗口内"
        intervals [2]:
            xmin = 3
            xmax = 4
            text = "窗口外"
''',
        encoding="utf-8",
    )

    module.validate_near_gold(case, "N_SPK1", textgrid)
