from __future__ import annotations

from collections import defaultdict
import json
from statistics import median
from typing import Any
import unicodedata


def contains_all(text: str, needles: list[str]) -> bool:
    normalized_text = normalize_match_text(text)
    return all(normalize_match_text(str(needle)) in normalized_text for needle in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    normalized_text = normalize_match_text(text)
    return any(normalize_match_text(str(needle)) in normalized_text for needle in needles)


# 评测匹配统一做 Unicode/大小写/空白归一，减少中文和 H2O 变体误判。
def normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return "".join(normalized.lower().split())


# 支持 debug_equals 这类 dotted path 断言。
def get_path(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


# 单轮硬判只看回复、召回、新增记忆和 debug，不让 LLM 自评参与打分。
def evaluate_turn(
    *,
    response: dict[str, Any] | None,
    exception: str,
    expect: dict[str, Any],
    new_memories: list[dict[str, Any]],
    all_memories: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": "" if passed else detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    if exception:
        add_check("no_exception", False, exception)
        return {"passed": False, "checks": checks, "failures": failures}

    response = response or {}
    reply = str(response.get("reply") or "")
    recalled_memories = [item for item in response.get("recalled_memories") or [] if isinstance(item, dict)]
    recalled_documents = [item for item in response.get("recalled_documents") or [] if isinstance(item, dict)]
    recalled_timeline_chunks = [
        item for item in response.get("recalled_timeline_chunks") or [] if isinstance(item, dict)
    ]
    recalled_text = "\n".join(str(item.get("content") or "") for item in recalled_memories)
    recalled_document_text = "\n".join(
        _metadata_text(item, ("filename", "title", "summary", "source", "ingestion_id"))
        for item in recalled_documents
    )
    recalled_timeline_text = "\n".join(str(item.get("text") or "") for item in recalled_timeline_chunks)
    new_memory_text = "\n".join(str(item.get("content") or "") for item in new_memories)
    active_memory_text = "\n".join(str(item.get("content") or "") for item in all_memories)
    debug_text = json.dumps(response.get("debug") or {}, ensure_ascii=False, sort_keys=True)

    # 回复文本断言用于验证用户可见结果是否命中金标准。
    if "reply_contains" in expect:
        needles = _as_list(expect.get("reply_contains"))
        add_check("reply_contains", contains_all(reply, needles), f"missing one of {needles!r} in reply={reply!r}")
    if "reply_contains_any" in expect:
        needles = _as_list(expect.get("reply_contains_any"))
        add_check("reply_contains_any", contains_any(reply, needles), f"missing any of {needles!r} in reply={reply!r}")
    if "reply_not_contains" in expect:
        needles = _as_list(expect.get("reply_not_contains"))
        normalized_reply = normalize_match_text(reply)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_reply]
        add_check("reply_not_contains", not matched, f"forbidden terms {matched!r} in reply={reply!r}")
    if "reply_not_contains_any" in expect:
        needles = _as_list(expect.get("reply_not_contains_any"))
        normalized_reply = normalize_match_text(reply)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_reply]
        add_check("reply_not_contains_any", not matched, f"forbidden terms {matched!r} in reply={reply!r}")

    # 召回断言单独检查，区分“答对了”和“确实从记忆中取到了证据”。
    if "recalled_contains" in expect:
        needles = _as_list(expect.get("recalled_contains"))
        add_check(
            "recalled_contains",
            contains_all(recalled_text, needles),
            f"missing one of {needles!r} in recalled={recalled_text!r}",
        )
    if "recalled_not_contains" in expect:
        needles = _as_list(expect.get("recalled_not_contains"))
        normalized_recalled = normalize_match_text(recalled_text)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_recalled]
        add_check("recalled_not_contains", not matched, f"forbidden terms {matched!r} in recalled={recalled_text!r}")
    if "recalled_memory_type" in expect:
        expected_types = {normalize_match_text(item) for item in _as_list(expect.get("recalled_memory_type"))}
        actual_types = {normalize_match_text(str(item.get("memory_type") or "")) for item in recalled_memories}
        add_check(
            "recalled_memory_type",
            bool(expected_types & actual_types),
            f"expected any of {sorted(expected_types)!r}, got {sorted(actual_types)!r}",
        )
    if "recalled_not_memory_type" in expect:
        forbidden_types = {normalize_match_text(item) for item in _as_list(expect.get("recalled_not_memory_type"))}
        matched_types = [
            str(item.get("memory_type") or "")
            for item in recalled_memories
            if normalize_match_text(str(item.get("memory_type") or "")) in forbidden_types
        ]
        add_check(
            "recalled_not_memory_type",
            not matched_types,
            f"forbidden memory_type values {matched_types!r} in recalled memories",
        )
    if "recalled_document_contains" in expect:
        needles = _as_list(expect.get("recalled_document_contains"))
        add_check(
            "recalled_document_contains",
            contains_all(recalled_document_text, needles),
            f"missing one of {needles!r} in recalled_documents={recalled_document_text!r}",
        )
    if "recalled_timeline_contains" in expect:
        needles = _as_list(expect.get("recalled_timeline_contains"))
        add_check(
            "recalled_timeline_contains",
            contains_all(recalled_timeline_text, needles),
            f"missing one of {needles!r} in recalled_timeline_chunks={recalled_timeline_text!r}",
        )
    if "recalled_timeline_not_contains" in expect:
        needles = _as_list(expect.get("recalled_timeline_not_contains"))
        normalized_timeline = normalize_match_text(recalled_timeline_text)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_timeline]
        add_check(
            "recalled_timeline_not_contains",
            not matched,
            f"forbidden terms {matched!r} in recalled_timeline_chunks={recalled_timeline_text!r}",
        )

    # 写入断言只看本轮新增记忆，避免旧记忆掩盖写入失败。
    if "saved_contains" in expect:
        needles = _as_list(expect.get("saved_contains"))
        add_check(
            "saved_contains",
            contains_all(new_memory_text, needles),
            f"missing one of {needles!r} in new memories={new_memory_text!r}",
        )
    if "saved_not_contains" in expect:
        needles = _as_list(expect.get("saved_not_contains"))
        normalized_new_memory = normalize_match_text(new_memory_text)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_new_memory]
        add_check("saved_not_contains", not matched, f"forbidden terms {matched!r} in new memories={new_memory_text!r}")
    if expect.get("no_saved"):
        add_check("no_saved", len(new_memories) == 0, f"new memories={new_memory_text!r}")
    if "saved_count" in expect:
        expected_count = int(expect.get("saved_count") or 0)
        add_check("saved_count", len(new_memories) == expected_count, f"expected {expected_count}, got {len(new_memories)}")
    if "saved_count_min" in expect:
        expected_count = int(expect.get("saved_count_min") or 0)
        add_check("saved_count_min", len(new_memories) >= expected_count, f"expected at least {expected_count}, got {len(new_memories)}")
    if "saved_kind" in expect and new_memories:
        expected_kind = str(expect.get("saved_kind"))
        memories_for_kind = _primary_saved_memories_for_kind_check(new_memories, expect)
        bad = [item for item in memories_for_kind if item.get("kind") != expected_kind]
        add_check("saved_kind", not bad, f"expected kind={expected_kind}, bad={bad!r}")
    if "active_memory_contains" in expect:
        needles = _as_list(expect.get("active_memory_contains"))
        add_check(
            "active_memory_contains",
            contains_all(active_memory_text, needles),
            f"missing one of {needles!r} in active memories={active_memory_text!r}",
        )
    if "active_memory_not_contains" in expect:
        needles = _as_list(expect.get("active_memory_not_contains"))
        normalized_active_memory = normalize_match_text(active_memory_text)
        matched = [needle for needle in needles if normalize_match_text(str(needle)) in normalized_active_memory]
        add_check(
            "active_memory_not_contains",
            not matched,
            f"forbidden terms {matched!r} in active memories={active_memory_text!r}",
        )
    if "active_memory_count" in expect:
        expected_count = int(expect.get("active_memory_count") or 0)
        add_check(
            "active_memory_count",
            len(all_memories) == expected_count,
            f"expected {expected_count}, got {len(all_memories)}",
        )

    # 主动提醒类 target 场景使用 triggered/not_triggered 表达触发结果。
    if "triggered" in expect:
        actual = bool(response.get("triggered"))
        expected = bool(expect.get("triggered"))
        add_check("triggered", actual == expected, f"expected {expected!r}, got {actual!r}")
    if expect.get("not_triggered"):
        add_check("not_triggered", not bool(response.get("triggered")), f"got triggered={response.get('triggered')!r}")

    if "api_calls" in expect:
        add_check("api_calls", response.get("api_calls") == expect.get("api_calls"), f"got {response.get('api_calls')!r}")
    if "memory_processing_status" in expect:
        status = get_path(response, "debug.memory_processing.status")
        expected_status = expect.get("memory_processing_status")
        add_check("memory_processing_status", status == expected_status, f"expected {expected_status!r}, got {status!r}")
    if "debug_contains" in expect:
        needles = _as_list(expect.get("debug_contains"))
        add_check(
            "debug_contains",
            contains_all(debug_text, needles),
            f"missing one of {needles!r} in debug={debug_text!r}",
        )
    if "audit_record_type" in expect:
        audit_record = get_path(response, "debug.eval_action.audit_record")
        actual_type = str(audit_record.get("record_type") or "") if isinstance(audit_record, dict) else ""
        expected_type = str(expect.get("audit_record_type") or "")
        add_check("audit_record_type", actual_type == expected_type, f"expected {expected_type!r}, got {actual_type!r}")
    for path, expected in dict(expect.get("debug_equals") or {}).items():
        actual = get_path(response, path)
        add_check(f"debug_equals:{path}", actual == expected, f"expected {expected!r}, got {actual!r}")
    for path, expected in dict(expect.get("debug_path_contains") or {}).items():
        expected_values = _as_value_list(expected)
        actual_values = _path_values(response, path)
        missing = [value for value in expected_values if value not in actual_values]
        add_check(
            f"debug_path_contains:{path}",
            not missing,
            f"missing {missing!r}, got {actual_values!r}",
        )

    return {"passed": not failures, "checks": checks, "failures": failures}


# 汇总时 active 和 target 分开统计，target 缺口不阻断当前门禁。
def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    turn_results = [turn for run in runs for turn in run.get("turns", [])]
    active_turns = [turn for turn in turn_results if turn.get("status", "active") != "target"]
    target_turns = [turn for turn in turn_results if turn.get("status") == "target"]
    total_turns = len(turn_results)
    passed_turns = sum(1 for turn in turn_results if turn.get("passed"))
    active_total_turns = len(active_turns)
    active_passed_turns = sum(1 for turn in active_turns if turn.get("passed"))
    target_passed_turns = sum(1 for turn in target_turns if turn.get("passed"))
    exceptions = sum(1 for turn in turn_results if turn.get("exception"))
    completed = sum(1 for turn in turn_results if (turn.get("response") or {}).get("completed") is True)

    expected_reply_checks = _checks_by_name(turn_results, ("reply_contains", "reply_contains_any", "reply_not_contains", "reply_not_contains_any"))
    expected_recall_checks = _checks_by_name(turn_results, (
        "recalled_contains",
        "recalled_not_contains",
        "recalled_memory_type",
        "recalled_not_memory_type",
        "recalled_document_contains",
        "recalled_timeline_contains",
    ))
    expected_save_checks = _checks_by_name(turn_results, (
        "saved_contains",
        "saved_not_contains",
        "no_saved",
        "saved_count",
        "saved_count_min",
        "saved_kind",
        "active_memory_contains",
        "active_memory_not_contains",
        "active_memory_count",
    ))
    isolation_turns = [turn for run in runs if run.get("category") == "isolation" for turn in run.get("turns", [])]

    timings = [_turn_total_seconds(turn) for turn in turn_results]
    timings = [value for value in timings if value is not None]
    stage_timings = _stage_timings(turn_results)
    api_calls = [(turn.get("response") or {}).get("api_calls") for turn in turn_results]
    api_calls = [int(value) for value in api_calls if isinstance(value, int)]

    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        scenario_groups[str(run.get("scenario_id"))].append(run)
    stable_groups = 0
    repeat_groups = 0
    for grouped in scenario_groups.values():
        if len(grouped) <= 1:
            continue
        repeat_groups += 1
        if all(run.get("passed") for run in grouped):
            stable_groups += 1

    return {
        "total_runs": len(runs),
        "total_turns": total_turns,
        "passed_turns": passed_turns,
        "failed_turns": total_turns - passed_turns,
        "pass_rate": _rate(passed_turns, total_turns),
        "active_turns": active_total_turns,
        "active_failed_turns": active_total_turns - active_passed_turns,
        "active_pass_rate": _rate(active_passed_turns, active_total_turns),
        "target_turns": len(target_turns),
        "target_failed_turns": len(target_turns) - target_passed_turns,
        "target_pass_rate": _rate(target_passed_turns, len(target_turns)),
        "exception_rate": _rate(exceptions, total_turns),
        "completed_rate": _rate(completed, total_turns),
        "reply_fact_hit_rate": _check_rate(expected_reply_checks),
        "memory_recall_hit_rate": _check_rate(expected_recall_checks),
        "memory_write_hit_rate": _check_rate(expected_save_checks),
        "isolation_pass_rate": _rate(sum(1 for turn in isolation_turns if turn.get("passed")), len(isolation_turns)),
        "repeat_stability_rate": _rate(stable_groups, repeat_groups),
        "api_calls_total": sum(api_calls),
        "api_calls_mean": round(sum(api_calls) / len(api_calls), 4) if api_calls else 0.0,
        "latency_seconds": _latency_summary(timings),
        "stage_latency_seconds": {name: _latency_summary(values) for name, values in stage_timings.items()},
        "slowest_turns": _slowest_turns(turn_results, limit=10),
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_value_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _path_values(payload: Any, dotted: str) -> list[Any]:
    values: list[Any] = [payload]
    for part in dotted.split("."):
        next_values: list[Any] = []
        for value in values:
            if part == "*":
                if isinstance(value, list):
                    next_values.extend(value)
                continue
            if isinstance(value, dict) and part in value:
                next_values.append(value[part])
        values = next_values
        if not values:
            return []
    return values


def _metadata_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(str(payload.get(key) or "") for key in keys)


def _primary_saved_memories_for_kind_check(
    new_memories: list[dict[str, Any]],
    expect: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = [
        item for item in new_memories
        if item.get("memory_type") != "observation"
        and item.get("source") != "observation_reflect"
    ]
    needles = _as_list(expect.get("saved_contains")) if "saved_contains" in expect else []
    if needles:
        matching_primary = [
            item for item in primary
            if contains_all(str(item.get("content") or ""), needles)
        ]
        if matching_primary:
            return matching_primary
    return primary or new_memories


def _checks_by_name(turns: list[dict[str, Any]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        check
        for turn in turns
        for check in turn.get("checks", [])
        if any(str(check.get("name", "")).startswith(name) for name in names)
    ]


def _check_rate(checks: list[dict[str, Any]]) -> float:
    return _rate(sum(1 for check in checks if check.get("passed")), len(checks))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _turn_total_seconds(turn: dict[str, Any]) -> float | None:
    response = turn.get("response") or {}
    debug = response.get("debug") or {}
    timing = debug.get("timing") or {}
    value = timing.get("total_seconds")
    if isinstance(value, (int, float)):
        return float(value)
    measured = turn.get("measured_seconds")
    return float(measured) if isinstance(measured, (int, float)) else None


# 阶段耗时从 debug.timing.stages 收集，用于定位慢在 planner、web 还是主模型。
def _stage_timings(turns: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for turn in turns:
        response = turn.get("response") or {}
        timing = ((response.get("debug") or {}).get("timing") or {})
        for stage in timing.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            seconds = stage.get("seconds")
            if isinstance(seconds, (int, float)):
                values[str(stage.get("name") or "unknown")].append(float(seconds))
    return values


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "p50": round(float(median(ordered)), 6),
        "p95": round(float(ordered[p95_index]), 6),
        "max": round(float(max(ordered)), 6),
    }


# 报告保留最慢样本，方便下一轮直接从具体输入复盘。
def _slowest_turns(turns: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for turn in turns:
        seconds = _turn_total_seconds(turn)
        if seconds is None:
            continue
        rows.append({
            "scenario_id": turn.get("scenario_id"),
            "repeat_index": turn.get("repeat_index"),
            "turn_index": turn.get("turn_index"),
            "message": turn.get("message"),
            "seconds": round(seconds, 6),
            "passed": turn.get("passed"),
        })
    return sorted(rows, key=lambda item: item["seconds"], reverse=True)[:limit]
