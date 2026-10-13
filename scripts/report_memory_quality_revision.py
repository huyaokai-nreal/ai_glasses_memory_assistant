"""Verify frozen evidence and compare against archived answers; never call a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import run_memory_quality_revision as revision
import run_live_memory_closeout as live


def read(path):
    return json.loads(path.read_text())


def contains_id(values, targets):
    return any((value.get("id") if isinstance(value, dict) else value) in targets for value in values)


def chains(run, protocol, ledger):
    output = []
    for case in protocol["conditions"]:
        path = run / "quality" / case["id"] / "after.json"
        row = dict(id=case["id"], arm="after", stage=case["stage"], condition=case["condition"],
                   group=case["group"], executed=path.exists())
        if path.exists():
            raw = read(path)
            result = raw.get("result", {})
            debug = result.get("debug", {})
            ppd = debug.get("pre_reply_decision", {})
            memory = debug.get("memory", {})
            event = memory.get("event_recall", {})
            targets = raw.get("seed", {}).get("target_ids", [])
            candidate = event.get("candidate_trace", {}).get("selected_ids") or event.get("ranking", [])
            candidate_hit = contains_id(candidate[:5], targets) or contains_id(memory.get("profile_memories", [])[:5], targets)
            final_hit = contains_id(result.get("recalled_memories", [])[:5], targets)
            gate = bool(ppd.get("needs_event_memory") or ppd.get("needs_profile_memory"))
            reserved = [r for r in ledger.rows if r["type"] == "reserve" and r["variant"] == f'quality/{case["id"]}/after']
            contexts = []
            for request in reserved:
                messages = request.get("request", {}).get("messages", [])
                system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
                reader = system.startswith("You are a stage-1 AI glasses personal memory assistant.") or "evidence-accounting Reader" in system
                if reader and case.get("content") and case["content"] in "\n".join(str(m.get("content", "")) for m in messages):
                    contexts.append(request["request_id"])
            reason = "not_target_case"
            if case.get("expected_target"):
                reason = ("execution_failed" if raw.get("error") else "gate_missed" if not gate else
                          "candidate_missed" if not candidate_hit else "context_lost" if not contexts else "answer_review_required")
            row.update(question=case["question"], reply=result.get("reply", ""), error=raw.get("error"),
                gate_open=gate, queries=ppd.get("memory_search_queries", []), target_ids=targets,
                candidate_recall_at5=candidate_hit, final_recall_at5=final_hit, context_request_ids=contexts,
                failure_layer=reason, generation_requests=len(reserved), arbitration=memory.get("recall_arbitration", {}),
                temporal=debug.get("temporal", {}), complete_set_answer=debug.get("complete_set_answer", {}),
                evidence=str(path.relative_to(run)), evidence_sha256=live.digest(path))
        output.append(row)
    return output


def summary(rows, reviews):
    reviewed = {r["id"]: r for r in reviews}
    positives = [r for r in rows if r["condition"] in {"zero_overlap", "paraphrase"}]
    zeros = [r for r in positives if r["condition"] == "zero_overlap"]
    holdout = [r for r in positives if r["group"] == "holdout"]
    originals = [r for r in rows if r["condition"] == "original_correct"]
    negatives = [r for r in rows if r["condition"] == "no_evidence"]
    times = [r for r in rows if r["stage"] == "time"]
    def hits(values, key): return sum(r.get(key) is True for r in values)
    def answers(values): return sum(reviewed.get(r["id"], {}).get("passed") is True for r in values)
    failures = Counter()
    for row in positives:
        if reviewed.get(row["id"], {}).get("passed") is not True:
            layer = row.get("failure_layer", "not_executed")
            failures["answer_use_failed" if layer == "answer_review_required" else layer] += 1
    gate_pass = hits(positives, "candidate_recall_at5") >= 11 and hits(zeros, "candidate_recall_at5") >= 7 and hits(holdout, "candidate_recall_at5") >= 5
    quality_pass = gate_pass and answers(originals) == 2 and answers(negatives) == 2 and answers(times) == 8
    return dict(executed=sum(r["executed"] for r in rows), denominator=24, reviewed=len(reviews),
        candidate=dict(positive=hits(positives, "candidate_recall_at5"), positive_denominator=12,
                       zero=hits(zeros, "candidate_recall_at5"), zero_denominator=8,
                       holdout=hits(holdout, "candidate_recall_at5"), holdout_denominator=6),
        final_recall=dict(positive=hits(positives, "final_recall_at5"), positive_denominator=12,
                          zero=hits(zeros, "final_recall_at5"), zero_denominator=8,
                          holdout=hits(holdout, "final_recall_at5"), holdout_denominator=6),
        positive_answers=answers(positives), original_correct=answers(originals), negative_safe=answers(negatives),
        time_pass=answers(times), time_denominator=8, failures_by_layer=dict(failures),
        passed=quality_pass and len(reviews) == 24 and all(r["executed"] for r in rows),
        product_quality_complete=quality_pass and answers(positives) >= 11 and len(reviews) == 24,
        candidate_gate_passed=gate_pass)


def generate(run):
    protocol = revision.configure(run)
    revision.verify(run)
    ledger = revision.ledger(run, protocol)
    status = ledger.dispatch("/status", {})
    if status["pending"] or status["current"]:
        raise ValueError("Requests/variant not sealed; report refused")
    reservations = [r for r in ledger.rows if r["type"] == "reserve"]
    for r in reservations:
        assert r["variant"] in protocol["variants"] and r["client"] == {k:live.CONFIG[k] for k in ["provider", "model", "base_url"]}
        assert r["request"]["model"] == live.CONFIG["model"]
    rows = chains(run, protocol, ledger)
    parent = live.ROOT / protocol["parent_run"]
    baseline = [r for r in read(parent / "quality-chain.json") if r["arm"] == "after"]
    baseline_reviews = [r for r in read(parent / "answer-review.json") if r["arm"] == "after"]
    for review in baseline_reviews:
        raw = read(parent / review["evidence"])
        assert hashlib.sha256(raw["result"]["reply"].encode()).hexdigest() == review["answer_sha256"]
    reviews = read(run / "answer-review.json") if (run / "answer-review.json").exists() else []
    assert len({r["id"] for r in reviews}) == len(reviews)
    assert {r["id"] for r in reviews} <= {c["id"] for c in protocol["conditions"]}
    for review in reviews:
        row = next(r for r in rows if r["id"] == review["id"])
        assert row["executed"] and row["evidence"] == review["evidence"]
        assert hashlib.sha256(row["reply"].encode()).hexdigest() == review["answer_sha256"]
        assert review["review_method"] == "manual event/person/date/provenance review; not string match"
    before, after = summary(baseline, baseline_reviews), summary(rows, reviews)
    metrics = dict(baseline=before, after=after, requests=status, old_results_rerun=False,
        case_freeze="previously inspected, not blind", real_database_installed=False, android_web_resumed=False,
        speed_gain_attributable=False, overall_complete=False)
    directory = run / "reports" / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    live.dump(directory / "quality-chain.json", rows)
    live.dump(directory / "metrics.json", metrics)
    manifest = {str(p.relative_to(live.ROOT)):live.digest(p) for p in [
        run / "protocol.json", run / "conditions.json", run / "freeze.json", run / "requests.jsonl",
        parent / "requests.jsonl", parent / "quality-chain.json", parent / "answer-review.json",
        run / "sources" / "after.json", run / "runner-hashes.json",
    ]}
    for row in rows:
        if row["executed"]: manifest[str((run / row["evidence"]).relative_to(live.ROOT))] = row["evidence_sha256"]
    if reviews: manifest[str((run / "answer-review.json").relative_to(live.ROOT))] = live.digest(run / "answer-review.json")
    live.dump(directory / "evidence-manifest.json", manifest)
    lines = ["# 召回与时间修复验收", "", "## 用户问题与修改", "",
        "本轮修复服务时区与日历目标消费、完整清单与数量分离、未知日期证据说明，并改进同次PPD的召回门控和检索改述。",
        "保留三条128字符改述和原候选预算；没有向量模型、额外判断调用或真实库安装。", "",
        "## 同口径前后指标", "",
        "旧after结果封存，本轮只运行新版本一次。以下历史留出组已被查看，不是盲测。", "",
        "| 指标 | 检查点版本 | 本轮版本 |", "|---|---|---|"]
    for label, values in [
        ("候选正例 /12", (before["candidate"]["positive"], after["candidate"]["positive"])),
        ("候选零词重合 /8", (before["candidate"]["zero"], after["candidate"]["zero"])),
        ("候选历史留出 /6", (before["candidate"]["holdout"], after["candidate"]["holdout"])),
        ("最终召回 /12", (before["final_recall"]["positive"], after["final_recall"]["positive"])),
        ("正确正例回答 /12", (before["positive_answers"], after["positive_answers"])),
        ("时间严格条件 /8", (before["time_pass"], after["time_pass"])),
        ("原正确保持 /2", (before["original_correct"], after["original_correct"])),
        ("无证据负例安全 /2", (before["negative_safe"], after["negative_safe"])),
    ]: lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", f'本轮执行 {after["executed"]}/24、人工复核 {after["reviewed"]}/24；冻结质量门槛通过={after["passed"]}。',
        f'新请求 {status["local_total"]}/120；累计 {status["total"]}/640，旧261次及旧1次结果未知保守计入，待结束{len(status["pending"])}。',
        "", "## 失败与退化", "", f'正例失败分层：{after["failures_by_layer"]}。']
    for review in reviews:
        if not review["passed"]: lines.append(f'- {review["id"]}：{review["reason"]}')
    regressions = [r["id"] for r in reviews if not r["passed"] and next(b for b in baseline_reviews if b["id"] == r["id"])["passed"]]
    lines += [f"对照已正确而本轮失败：{regressions}。", "", "## 证据边界", "",
        "逐例答案由事件、人物、日期及证据关系人工复核，不把字符串出现或固定正确合同当作PPD质量。",
        "语义改述仍是有界词法检索；未覆盖任意改述。离线检查不能替代真实模型或手机体验。",
        "Android/Web继续停止，真实库未安装，速度未验；本期没有LongMemEval或新音频验收数字。", "",
        "## 下一步", "",
        "门槛失败或出现退化时保留未完成、封存该版本，不挑例、不补跑、不改门槛。",
        "通过的局部修复可交付，但整体完成仍需要独立的产品落地与体验证据。"]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    print(directory)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    generate(parser.parse_args().run.resolve())
