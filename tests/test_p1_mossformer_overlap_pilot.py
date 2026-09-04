from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "p1_mossformer_overlap_pilot.py"
    spec = importlib.util.spec_from_file_location("p1_mossformer_overlap_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_permutation_cer_chooses_speaker_order() -> None:
    module = _module()

    result = module.permutation_cer(("甲乙", "丙丁"), ("丙丁", "甲乙"))

    assert result["cer"] == 0.0
    assert result["hypotheses"] == ["甲乙", "丙丁"]


def test_aggregate_results_compares_two_stream_candidate_with_raw() -> None:
    module = _module()
    rows = [
        {
            "duration_s": 1.0,
            "realtime_factor": 2.0,
            "raw_single_stream": {"errors": 4, "reference_chars": 10},
            "mossformer2_two_stream": {"errors": 2, "reference_chars": 10},
            "near_oracle_two_stream": {"errors": 1, "reference_chars": 10},
        },
        {
            "duration_s": 2.0,
            "realtime_factor": 4.0,
            "raw_single_stream": {"errors": 3, "reference_chars": 10},
            "mossformer2_two_stream": {"errors": 5, "reference_chars": 10},
            "near_oracle_two_stream": {"errors": 0, "reference_chars": 10},
        },
    ]

    result = module.aggregate_results(rows)

    assert result["raw_single_stream"]["cer"] == 0.35
    assert result["mossformer2_two_stream"]["cer"] == 0.35
    assert result["near_oracle_two_stream"]["cer"] == 0.05
    assert result["clip_outcomes_vs_raw"] == {"improved": 1, "equal": 0, "worse": 1}
    assert result["realtime_factor"] == {"mean": 3.0, "median": 3.0, "max": 4.0}
