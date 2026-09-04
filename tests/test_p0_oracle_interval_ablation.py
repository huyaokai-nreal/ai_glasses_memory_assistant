from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ai_glasses_memory_assistant.evals.eval_ali import EvalAliCase, ReferenceInterval


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "p0_oracle_interval_ablation.py"
    spec = importlib.util.spec_from_file_location("p0_oracle_interval_ablation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_overlap_split_and_aggregate_preserve_deletion_counts() -> None:
    module = _module()
    first = ReferenceInterval("a", 1.0, 2.0, "甲乙")
    second = ReferenceInterval("b", 1.5, 2.5, "丙丁")
    clean = ReferenceInterval("a", 3.0, 4.0, "戊己")
    intervals = (first, second, clean)
    case = EvalAliCase("case", "session", "audio", "grid", "a", "b", 5.0, 0.0, 5.0, intervals)

    assert module.overlaps_other_speaker(first, intervals) is True
    assert module.overlaps_other_speaker(clean, intervals) is False
    overlap_row = module.score_interval(case=case, interval=first, all_intervals=intervals, hypothesis="甲")
    clean_row = module.score_interval(case=case, interval=clean, all_intervals=intervals, hypothesis="戊己")
    total = module.aggregate([overlap_row, clean_row])

    assert overlap_row.overlap_exposed is True
    assert clean_row.overlap_exposed is False
    assert total["reference_chars"] == 4
    assert total["deletions"] == 1
    assert total["normalized_cer"] == 0.25
    assert total["shares_of_errors"]["deletion"] == 1.0
    assert total["intervals"] == 2
