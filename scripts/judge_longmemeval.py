"""LongMemEval official judge — GPT semantic equivalence scorer.

Uses the LLM configured via LLM_PROVIDER/LLM_MODEL/LLM_BASE_URL/LLM_API_KEY env vars,
with a fallback to DEEPSEEK_FALLBACK_PROVIDER.

Usage:
    conda run -n hermes python scripts/judge_longmemeval.py \
        reports/longmemeval/oracle-preference-20260804_170941/longmemeval_oracle_memory.jsonl \
        .external/voice-recording/scripts/memory/longmemeval_oracle.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from ai_glasses_memory_assistant.env_loader import load_app_dotenv
from ai_glasses_memory_assistant.evals.longmemeval_runner import render_markdown
from ai_glasses_memory_assistant.llm_runtime import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_FALLBACK_PROVIDER,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_PROVIDER_ENV,
)


JUDGE_SYSTEM = """You are an answer equivalence judge for the LongMemEval benchmark.
Compare the hypothesis with the reference answer and decide if they are semantically equivalent.

Scoring rules:
- 1 (CORRECT): The hypothesis and reference answer convey the SAME meaning, even if wording differs.
  - Preferences, recommendations, constraints: if the hypothesis captures the core preference/constraint, it's correct.
  - Names, dates, numbers: must match exactly or be semantically equivalent.
  - Counts/comparisons: the conclusion must match.
- 0 (INCORRECT): The hypothesis is missing, contradictory, too vague, or refuses to answer ("I don't know...").
  - A refusal or "no information" answer is ALWAYS incorrect.
  - A partially correct but substantially incomplete answer is incorrect.

Return JSON only:
{"score": 0 or 1, "reason": "<brief explanation in English>"}"""


def _parse_json_robust(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.insert(0, raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Try repairing common issues: unescaped quotes, trailing commas
            for repair in (
                candidate.replace("\t", " "),
                candidate.replace("\n", " "),
            ):
                try:
                    parsed = json.loads(repair)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if isinstance(parsed, dict) and "score" in parsed:
            return parsed
    return None


def judge_single(client, model, question: str, reference: str, hypothesis: str, *, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            user_content = (
                f"Question: {question}\n\n"
                f"Reference answer: {reference}\n\n"
                f"Hypothesis: {hypothesis}\n\n"
                f"Is the hypothesis semantically equivalent to the reference answer? Return JSON."
            )
            if attempt > 0:
                # 空/截断响应后追加一次"只输出 JSON"指令；不改评分规则。
                user_content += '\n\nReturn JSON only with {"score": 0 or 1, "reason": "..."}.'
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
        except Exception:
            time.sleep(2)
            continue
        parsed = _parse_json_robust(content)
        if parsed is not None:
            return {"score": int(parsed.get("score", 0)), "reason": str(parsed.get("reason", ""))[:200]}
        # Empty/truncated responses are transient; retry.
        time.sleep(1.5)
    return {"score": 0, "reason": "judge parse error"}


def _judge_all(
    client,
    model: str,
    hypotheses: dict[str, str],
    oracle_items: list[dict],
) -> tuple[dict[str, int], dict[str, str], int]:
    """Return (per_question_score, per_question_reason, refusals)."""
    per_question: dict[str, int] = {}
    reasons: dict[str, str] = {}
    refusals = 0
    for item in oracle_items:
        qid = item["question_id"]
        if qid not in hypotheses:
            continue
        hypothesis = hypotheses[qid]
        if "I don't know based on the available memory" in hypothesis:
            refusals += 1
            per_question[qid] = 0
            reasons[qid] = "refusal"
            print(f"  {qid} → 0 (refusal)")
            continue
        result = judge_single(client, model, item["question"], item["answer"], hypothesis)
        per_question[qid] = result["score"]
        reasons[qid] = result["reason"]
        status = "✓" if result["score"] else "✗"
        print(f"  {qid} → {result['score']} {status}  ({result['reason'][:80]})")
        time.sleep(0.3)  # rate limit
    return per_question, reasons, refusals


def _summarize_judge(per_question: dict[str, int], refusals: int, reasons: dict[str, str] | None = None) -> dict:
    total = len(per_question)
    correct = sum(per_question.values())
    reasons = reasons or {}
    # parse error means the judge could not produce a JSON verdict (empty or
    # truncated response after retries). It is NOT a "wrong" verdict: the
    # hypothesis may still be correct, it simply was not evaluated.
    parse_errors = sum(1 for qid in per_question if reasons.get(qid) == "judge parse error")
    # genuine wrong verdicts: judged as 0 and the judge parsed successfully.
    wrong = sum(1 for qid, score in per_question.items() if score == 0 and reasons.get(qid) != "judge parse error")
    return {
        "total": total,
        "correct": correct,
        "correct_rate": round(correct / total, 4) if total else 0.0,
        "refusals": refusals,
        "parse_errors": parse_errors,
        "wrong": wrong,
        # Conservative headline: parse errors count as 0 in the denominator,
        # but are listed separately so they are not misread as judged-wrong.
        "correct_excl_refusals": correct,
        "correct_excl_refusals_rate": round(correct / (total - refusals), 4) if total > refusals else 0.0,
    }


def _make_client() -> tuple[Any, str]:
    load_app_dotenv()
    provider = os.environ.get(LLM_PROVIDER_ENV, DEEPSEEK_FALLBACK_PROVIDER)
    model = os.environ.get(LLM_MODEL_ENV, "deepseek-chat")
    base_url = os.environ.get(LLM_BASE_URL_ENV, "")
    api_key = os.environ.get(LLM_API_KEY_ENV, "") or os.environ.get(DEEPSEEK_API_KEY_ENV, "")

    if not base_url:
        if provider.lower() == "deepseek":
            base_url = "https://api.deepseek.com/v1"
        else:
            base_url = "https://api.openai.com/v1"
    if not api_key:
        print("Error: Set AI_GLASSES_LLM_API_KEY or DEEPSEEK_API_KEY in .env.")
        sys.exit(1)

    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url), model


def _load_hypotheses(path: Path) -> dict[str, str]:
    hypotheses: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            hypotheses[item["question_id"]] = item["hypothesis"]
    return hypotheses


def judge_report_dir(report_dir: Path, oracle_path: Path, *, judge_model: str = "") -> dict:
    """Run official judge on a report directory and write results back into
    eval-latest.json (summary.official_judge) and eval-latest.md."""
    report_dir = Path(report_dir)
    hypotheses_path = report_dir / "longmemeval_oracle_memory.jsonl"
    eval_path = report_dir / "eval-latest.json"
    if not hypotheses_path.exists():
        # fallback: any *memory.jsonl
        candidates = sorted(report_dir.glob("*memory.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No hypotheses jsonl under {report_dir}")
        hypotheses_path = candidates[0]
    client, model = _make_client()
    if judge_model:
        model = judge_model
    hypotheses = _load_hypotheses(hypotheses_path)
    with open(oracle_path) as f:
        oracle_items = json.load(f)
    print(f"Judging {len(hypotheses)} hypotheses from {report_dir} ...")
    per_question, reasons, refusals = _judge_all(client, model, hypotheses, oracle_items)
    official = {
        "judge_model": model,
        "judge_provider": os.environ.get(LLM_PROVIDER_ENV, DEEPSEEK_FALLBACK_PROVIDER),
        "overall": _summarize_judge(per_question, refusals, reasons),
        "per_question": {qid: score for qid, score in sorted(per_question.items())},
        "reasons": reasons,
    }
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        payload["summary"]["official_judge"] = official
        payload["config"]["official_judge_model"] = model
        eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (report_dir / "eval-latest.md").write_text(render_markdown(payload), encoding="utf-8")
        print(f"Wrote official judge results back to {eval_path}")
    else:
        print("eval-latest.json not found; official judge results not persisted.")
    overall = official["overall"]
    print(f"\n{'='*40}")
    print(f"Total: {overall['total']}")
    print(f"Correct: {overall['correct']}")
    print(f"Score: {overall['correct_rate'] * 100:.1f}%")
    print(f"Refusals: {overall['refusals']}")
    print(f"Correct (excl refusals): {overall['correct']}/{overall['total'] - overall['refusals']} = {overall['correct_excl_refusals_rate'] * 100:.1f}%")
    return official


def main():
    argv = sys.argv[1:]
    if len(argv) >= 1 and argv[0] == "--report-dir":
        if len(argv) < 3:
            print(f"Usage: {sys.argv[0]} --report-dir <report_dir> <oracle_dataset.json>")
            sys.exit(1)
        judge_report_dir(argv[1], argv[2])
        return
    if len(argv) < 2:
        print(f"Usage: {sys.argv[0]} <hypotheses.jsonl> <oracle_dataset.json>")
        print(f"       {sys.argv[0]} --report-dir <report_dir> <oracle_dataset.json>")
        sys.exit(1)

    hypotheses_path = Path(argv[0])
    oracle_path = Path(argv[1])
    client, model = _make_client()
    hypotheses = _load_hypotheses(hypotheses_path)
    with open(oracle_path) as f:
        oracle_items = json.load(f)
    per_question, reasons, refusals = _judge_all(client, model, hypotheses, oracle_items)
    summary = _summarize_judge(per_question, refusals, reasons)
    print(f"\n{'='*40}")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']}")
    print(f"Score: {summary['correct_rate'] * 100:.1f}%")
    print(f"Refusals: {summary['refusals']}")
    print(f"Correct (excl refusals): {summary['correct']}/{summary['total'] - summary['refusals']} = {summary['correct_excl_refusals_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
