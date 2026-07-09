from __future__ import annotations

from typing import Any

from .conversation_helpers import ConversationSession
from .memory_candidate import MemoryWriteCandidate


def conversation_memory_candidates(
    session: ConversationSession,
) -> tuple[list[MemoryWriteCandidate], dict[str, Any]]:
    debug = session.debug_payload()
    debug["candidate_count"] = 0
    debug["candidate_strategy"] = "structural_only"
    debug["candidate_turn_indices"] = []
    debug["candidate_facts"] = []
    debug["saved_candidates"] = []
    debug["rejected_turns"] = []
    debug["rejected_reasons"] = []
    debug["candidate_previews"] = []
    return [], debug
