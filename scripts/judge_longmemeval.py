"""LongMemEval official QA-evaluation protocol with a configurable judge model.

Uses the LLM configured via LLM_PROVIDER/LLM_MODEL/LLM_BASE_URL/LLM_API_KEY env vars,
with a fallback to DEEPSEEK_FALLBACK_PROVIDER.

The prompts and yes/no decision rule mirror LongMemEval's upstream
``src/evaluation/evaluate_qa.py``.  Choosing a model other than the upstream
GPT-4o reference judge is a judge-model substitution and is recorded in the
report.

Usage:
    conda run -n hermes python scripts/judge_longmemeval.py \
        reports/longmemeval/oracle-preference-20260804_170941/longmemeval_oracle_memory.jsonl \
        .external/voice-recording/scripts/memory/longmemeval_oracle.json
"""

import json
import os
import sys
import time
from datetime import datetime
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


OFFICIAL_PROTOCOL_SOURCE = (
    "https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py"
)
OFFICIAL_REFERENCE_JUDGE = "gpt-4o-2024-08-06"
JUDGE_MAX_TOKENS_ENV = "LONGMEMEVAL_JUDGE_MAX_TOKENS"
JUDGE_THINKING_ENV = "AI_GLASSES_JUDGE_THINKING"
OFFICIAL_MAX_TOKENS = 10


def _judge_thinking_enabled(thinking: str | None = None) -> bool:
    raw = str(thinking if thinking is not None else os.environ.get(JUDGE_THINKING_ENV, "disabled"))
    return raw.strip().lower() == "enabled"


def _judge_thinking_extra_body(thinking: str | None = None) -> dict[str, Any] | None:
    """Disable provider thinking for the yes/no judge call by default.

    The official judge expects a bare "yes"/"no"; thinking mode burns tokens and
    adds latency without changing the verdict. ``--thinking enabled`` or
    ``AI_GLASSES_JUDGE_THINKING=enabled`` restores the default provider behavior.
    """
    if _judge_thinking_enabled(thinking):
        return None
    provider = os.environ.get(LLM_PROVIDER_ENV, DEEPSEEK_FALLBACK_PROVIDER).strip().lower()
    if provider == "deepseek":
        return {"thinking": {"type": "disabled"}}
    if provider == "llama_cpp":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def get_official_anscheck_prompt(
    question_type: str,
    question: str,
    reference: str,
    hypothesis: str,
    *,
    abstention: bool = False,
) -> str:
    """Build the task-specific prompt from LongMemEval's official evaluator."""
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: "
            "{}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    elif question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes "
            "if the response contains the correct answer. Otherwise, answer no. If the response is equivalent "
            "to the correct answer or contains all the intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
            "response correct? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes "
            "if the response contains the correct answer. Otherwise, answer no. If the response is equivalent "
            "to the correct answer or contains all the intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the number of days. If the question "
            "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
            "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: "
            "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response should be considered as correct "
            "as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        raise NotImplementedError(f"Unsupported LongMemEval question type: {question_type}")
    return template.format(question, reference, hypothesis)


def judge_single(
    client,
    model,
    question_type: str,
    question: str,
    reference: str,
    hypothesis: str,
    *,
    abstention: bool = False,
    retries: int = 5,
    max_tokens: int = OFFICIAL_MAX_TOKENS,
    extra_body: dict | None = None,
) -> dict:
    prompt = get_official_anscheck_prompt(
        question_type,
        question,
        reference,
        hypothesis,
        abstention=abstention,
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2)
            continue
        if content:
            # Match the upstream label rule exactly for a valid judge response.
            score = 1 if "yes" in content.lower() else 0
            return {"score": score, "reason": f"judge response: {content[:180]}"}
        # Empty responses are transport/provider failures, not model-wrong verdicts.
        time.sleep(1.5)
    return {"score": 0, "reason": "judge parse error"}


def _judge_all(
    client,
    model: str,
    hypotheses: dict[str, str],
    oracle_items: list[dict],
    *,
    max_tokens: int = OFFICIAL_MAX_TOKENS,
    extra_body: dict | None = None,
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
        result = judge_single(
            client,
            model,
            item["question_type"],
            item["question"],
            item["answer"],
            hypothesis,
            abstention="_abs" in qid,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
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
    }


def _summarize_by_question_type(per_question: dict[str, int], oracle_items: list[dict]) -> dict[str, dict]:
    qid_to_type = {item["question_id"]: item["question_type"] for item in oracle_items}
    grouped: dict[str, list[int]] = {}
    for qid, score in per_question.items():
        grouped.setdefault(qid_to_type[qid], []).append(score)
    return {
        question_type: {
            "total": len(scores),
            "correct": sum(scores),
            "correct_rate": round(sum(scores) / len(scores), 4),
        }
        for question_type, scores in sorted(grouped.items())
    }


def _summarize_by_abstention(per_question: dict[str, int]) -> dict[str, dict]:
    grouped = {
        "answerable": [score for qid, score in per_question.items() if "_abs" not in qid],
        "abstention": [score for qid, score in per_question.items() if "_abs" in qid],
    }
    return {
        group: {
            "total": len(scores),
            "correct": sum(scores),
            "correct_rate": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }
        for group, scores in grouped.items()
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


def judge_report_dir(report_dir: Path, oracle_path: Path, *, judge_model: str = "", thinking: str | None = None) -> dict:
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
    max_tokens = int(os.environ.get(JUDGE_MAX_TOKENS_ENV, str(OFFICIAL_MAX_TOKENS)))
    extra_body = _judge_thinking_extra_body(thinking)
    print(f"Judging {len(hypotheses)} hypotheses from {report_dir} ...")
    print(f"Judge model: {model}; max_tokens: {max_tokens}")
    per_question, reasons, refusals = _judge_all(
        client,
        model,
        hypotheses,
        oracle_items,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    official = {
        "judged_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge_model": model,
        "judge_provider": os.environ.get(LLM_PROVIDER_ENV, DEEPSEEK_FALLBACK_PROVIDER),
        "judge_thinking": "enabled" if extra_body is None else "disabled",
        "protocol": "longmemeval-official-qa-v1",
        "protocol_source": OFFICIAL_PROTOCOL_SOURCE,
        "upstream_reference_judge": OFFICIAL_REFERENCE_JUDGE,
        "judge_model_substitution": model != OFFICIAL_REFERENCE_JUDGE,
        "judge_max_tokens": max_tokens,
        "upstream_max_tokens": OFFICIAL_MAX_TOKENS,
        "overall": _summarize_judge(per_question, refusals, reasons),
        "by_question_type": _summarize_by_question_type(per_question, oracle_items),
        "by_abstention": _summarize_by_abstention(per_question),
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
    print(f"Canonical unknown responses (diagnostic only): {overall['refusals']}")
    return official


def main():
    argv = sys.argv[1:]
    thinking: str | None = None
    if "--thinking" in argv:
        index = argv.index("--thinking")
        if index + 1 < len(argv):
            thinking = argv[index + 1]
        argv = argv[:index] + argv[index + 2:]
    if len(argv) >= 1 and argv[0] == "--report-dir":
        if len(argv) < 3:
            print(f"Usage: {sys.argv[0]} --report-dir <report_dir> <oracle_dataset.json>")
            sys.exit(1)
        judge_report_dir(argv[1], argv[2], thinking=thinking)
        return
    if len(argv) < 2:
        print(f"Usage: {sys.argv[0]} <hypotheses.jsonl> <oracle_dataset.json>")
        print(f"       {sys.argv[0]} --report-dir <report_dir> <oracle_dataset.json> [--thinking enabled|disabled]")
        sys.exit(1)

    hypotheses_path = Path(argv[0])
    oracle_path = Path(argv[1])
    client, model = _make_client()
    hypotheses = _load_hypotheses(hypotheses_path)
    with open(oracle_path) as f:
        oracle_items = json.load(f)
    max_tokens = int(os.environ.get(JUDGE_MAX_TOKENS_ENV, str(OFFICIAL_MAX_TOKENS)))
    per_question, reasons, refusals = _judge_all(
        client,
        model,
        hypotheses,
        oracle_items,
        max_tokens=max_tokens,
        extra_body=_judge_thinking_extra_body(thinking),
    )
    summary = _summarize_judge(per_question, refusals, reasons)
    print(f"\n{'='*40}")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']}")
    print(f"Score: {summary['correct_rate'] * 100:.1f}%")
    print(f"Canonical unknown responses (diagnostic only): {summary['refusals']}")


if __name__ == "__main__":
    main()
