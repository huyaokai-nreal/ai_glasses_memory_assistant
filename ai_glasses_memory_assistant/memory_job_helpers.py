from __future__ import annotations

import uuid
from typing import Any

from .memory_kernel import source_trace


def create_memory_job_payload(
    *,
    user_id: str,
    session_id: str,
    mode: str,
    candidate_count: int,
    created_at: float,
    default_min_source_memory_count: int,
    source_memory_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    min_source_memory_count: int | None = None,
) -> dict[str, Any]:
    source_memory_ids = unique_strings(source_memory_ids or [])
    evidence_ids = unique_strings(evidence_ids or [])
    return {
        "job_id": f"memjob_{uuid.uuid4().hex[:16]}",
        "user_id": user_id,
        "session_id": session_id or "",
        "mode": mode,
        "status": "pending",
        "candidate_count": max(0, int(candidate_count)),
        "saved_count": 0,
        "rejected_count": 0,
        "saved_memory_ids": [],
        "rejected_reasons": [],
        "superseded_memory_ids": [],
        "superseded_observation_ids": [],
        "dedupe_decisions": [],
        "lifecycle_transitions": [],
        "task_status_updates": [],
        "task_status_policies": [],
        "correction_target_resolution": default_correction_target_resolution_payload(),
        "observation_update_decisions": [],
        "extraction_trace": {},
        "source_memory_count": len(source_memory_ids),
        "min_source_memory_count": (
            int(min_source_memory_count)
            if min_source_memory_count is not None
            else int(default_min_source_memory_count)
        ),
        "source_memory_ids": source_memory_ids,
        "evidence_ids": evidence_ids,
        "decision_reason": "",
        "skip_policy": {},
        "safety_policy": {},
        "question_policy": {},
        "confidence_policy": {},
        "error_type": "",
        "extraction_backend": "",
        "source_trace": source_trace(
            layer="reflection" if mode == "observation_reflect" else "structured_memory",
            user_id=user_id,
            source=mode,
            source_id="",
            evidence_ids=evidence_ids,
            status="pending",
        ),
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
    }


def public_memory_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    memory_processing = memory_job_processing_payload(job)
    return {
        "job_id": job["job_id"],
        "user_id": job["user_id"],
        "session_id": job.get("session_id", ""),
        "mode": job.get("mode", ""),
        "status": job.get("status", "pending"),
        "candidate_count": job.get("candidate_count", 0),
        "saved_count": job.get("saved_count", 0),
        "rejected_count": job.get("rejected_count", 0),
        "saved_memory_ids": list(job.get("saved_memory_ids") or []),
        "rejected_reasons": list(job.get("rejected_reasons") or []),
        "superseded_memory_ids": list(job.get("superseded_memory_ids") or []),
        "superseded_observation_ids": list(job.get("superseded_observation_ids") or []),
        "dedupe_decisions": [dict(item) for item in job.get("dedupe_decisions") or [] if isinstance(item, dict)],
        "lifecycle_transitions": [
            dict(item) for item in job.get("lifecycle_transitions") or [] if isinstance(item, dict)
        ],
        "task_status_updates": [
            dict(item) for item in job.get("task_status_updates") or [] if isinstance(item, dict)
        ],
        "task_status_policies": [
            dict(item) for item in job.get("task_status_policies") or [] if isinstance(item, dict)
        ],
        "correction_target_resolution": dict(
            job.get("correction_target_resolution") or default_correction_target_resolution_payload()
        ),
        "observation_update_decisions": [
            dict(item) for item in job.get("observation_update_decisions") or [] if isinstance(item, dict)
        ],
        "correction_detection": dict(job.get("correction_detection") or default_correction_detection_payload()),
        "extraction_trace": dict(job.get("extraction_trace") or {}),
        "source_memory_count": job.get("source_memory_count", 0),
        "source_memory_ids": list(job.get("source_memory_ids") or []),
        "evidence_ids": list(job.get("evidence_ids") or []),
        "decision_reason": job.get("decision_reason", ""),
        "skip_policy": dict(job.get("skip_policy") or {}),
        "safety_policy": dict(job.get("safety_policy") or {}),
        "question_policy": dict(job.get("question_policy") or {}),
        "confidence_policy": dict(job.get("confidence_policy") or {}),
        "error_type": job.get("error_type", ""),
        "extraction_backend": job.get("extraction_backend", ""),
        "source_trace": dict(job.get("source_trace") or {}),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "memory_processing": memory_processing,
    }


def memory_job_processing_payload(job: dict[str, Any]) -> dict[str, Any]:
    status = str(job.get("status") or "pending")
    extraction_trace = dict(job.get("extraction_trace") or {})
    stage_reason = memory_job_stage_reason(
        status=status,
        extraction_trace=extraction_trace,
        candidate_count_hint=int(job.get("candidate_count") or 0),
        rejected_reasons=list(job.get("rejected_reasons") or []),
        decision_reason=str(job.get("decision_reason") or ""),
        error_type=str(job.get("error_type") or ""),
    )
    payload = {
        "status": status,
        "stage": stage_reason.get("stage", ""),
        "stage_reason": stage_reason.get("reason", ""),
        "stage_explanation": stage_reason.get("explanation", ""),
        "saved_count": int(job.get("saved_count") or 0),
        "rejected_count": int(job.get("rejected_count") or 0),
        "decision_reason": str(job.get("decision_reason") or ""),
        "skip_policy": dict(job.get("skip_policy") or {}),
        "safety_policy": dict(job.get("safety_policy") or {}),
        "question_policy": dict(job.get("question_policy") or {}),
        "confidence_policy": dict(job.get("confidence_policy") or {}),
        "error_type": str(job.get("error_type") or ""),
    }
    local_do_not_remember_scopes = local_do_not_remember_scopes_from_trace(extraction_trace)
    if local_do_not_remember_scopes:
        payload["local_do_not_remember_scopes"] = local_do_not_remember_scopes
    if stage_reason.get("details"):
        payload["stage_details"] = dict(stage_reason.get("details") or {})
    return payload


def local_do_not_remember_scopes_from_trace(extraction_trace: dict[str, Any]) -> list[str]:
    semantic = extraction_trace.get("semantic_cleaning") if isinstance(extraction_trace, dict) else {}
    decisions = semantic.get("segment_decisions") if isinstance(semantic, dict) else []
    scopes: list[str] = []
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        scope = str(decision.get("do_not_remember_scope") or "").strip()
        if scope and scope not in scopes:
            scopes.append(scope)
    return scopes


def memory_job_stage_reason(
    *,
    status: str,
    extraction_trace: dict[str, Any],
    candidate_count_hint: int,
    rejected_reasons: list[str],
    decision_reason: str,
    error_type: str,
) -> dict[str, Any]:
    semantic = dict(extraction_trace.get("semantic_cleaning") or {})
    extraction = dict(extraction_trace.get("memory_extraction") or {})
    extractable_count = int(semantic.get("extractable_segment_count") or 0)
    skipped_count = int(semantic.get("skipped_segment_count") or 0)
    candidate_count = max(int(extraction.get("candidate_count") or 0), int(candidate_count_hint or 0))
    gate_rejected_count = int(extraction.get("gate_rejected_count") or 0)
    error_count = int(extraction.get("extraction_error_count") or 0)
    if status == "failed":
        if error_count > 0 and not error_type:
            return {
                "stage": "memory_extraction",
                "reason": "extractor_error",
                "explanation": "semantic cleaner 已放行，但 extractor 阶段出错了。",
            }
        if candidate_count > 0:
            return {
                "stage": "write_failure",
                "reason": f"write_exception:{error_type or 'unknown'}",
                "explanation": "候选已经通过前面阶段，但写入或后续流程失败了。",
            }
        return {
            "stage": "memory_extraction" if extractable_count > 0 else "semantic_cleaning",
            "reason": f"failed:{error_type or 'unknown'}",
            "explanation": "后台 memory job 失败了。",
        }
    if status == "saved":
        return {
            "stage": "saved",
            "reason": "memory_saved",
            "explanation": "候选通过抽取和门控后，已经成功写入。",
        }
    if status == "pending":
        return {
            "stage": "pending",
            "reason": "background_job_pending",
            "explanation": "后台 memory job 已创建，正在等待处理。",
        }
    if status == "running":
        return {
            "stage": "running",
            "reason": "background_job_running",
            "explanation": "后台 memory job 正在处理中。",
        }
    if status == "skipped":
        if skipped_count > 0 and candidate_count <= 0:
            return {
                "stage": "semantic_cleaning",
                "reason": "semantic_cleaner_rejected_all_segments",
                "explanation": "marker 虽然命中了，但 semantic cleaner 判断这些片段都不该抽取。",
                "details": {
                    "skipped_segment_count": skipped_count,
                    "extractable_segment_count": extractable_count,
                },
            }
        if extractable_count > 0 and candidate_count <= 0:
            return {
                "stage": "memory_extraction",
                "reason": "extractor_produced_no_candidates",
                "explanation": "semantic cleaner 已放行 span，但 extractor 没有产出可写入候选。",
                "details": {
                    "extractable_segment_count": extractable_count,
                    "candidate_count": candidate_count,
                },
            }
        return {
            "stage": "semantic_cleaning",
            "reason": decision_reason or "semantic_cleaner_rejected_all_segments",
            "explanation": "这次没有产生可保存的长期记忆，但在抽取前就被判定为无需继续处理。",
        }
    if status == "rejected":
        if gate_rejected_count > 0 or rejected_reasons:
            return {
                "stage": "gate",
                "reason": decision_reason or ",".join(rejected_reasons) or "write_gate_rejected_candidates",
                "explanation": "候选已经抽出来了，但被写入门控拒绝。",
                "details": {
                    "candidate_count": candidate_count,
                    "gate_rejected_count": gate_rejected_count,
                },
            }
        if extractable_count > 0 and candidate_count <= 0:
            return {
                "stage": "memory_extraction",
                "reason": "extractor_produced_no_candidates",
                "explanation": "semantic cleaner 已放行，但 extractor 没抽出候选。",
            }
        return {
            "stage": "semantic_cleaning",
            "reason": decision_reason or "semantic_cleaner_rejected_all_segments",
            "explanation": "这次被判定为不应写入长期记忆。",
        }
    return {
        "stage": status,
        "reason": decision_reason or status,
        "explanation": "memory job 已结束。",
    }


def default_correction_target_resolution_payload() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "matched_memory_ids": [],
        "superseded_memory_ids": [],
        "resolution_backend": "none",
        "fallback_reason": "",
    }


def default_correction_detection_payload() -> dict[str, Any]:
    return {
        "backend": "none",
        "matched_rule": "",
        "confidence": 0.0,
        "reason": "",
        "candidate_count": 0,
    }


def unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in values if str(item).strip()))
