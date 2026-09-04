# LongMemEval Goal: 80% First, 90% Only With Evidence

## Goal

Improve the current 500-case LongMemEval run at `reports/longmemeval/pref500-20260902-182616-35190` from the local-Qwen official-protocol baseline of 379/500 (75.8%) to at least 80% (at least 400/500), without benchmark-specific behavior, cloud calls, or regressions to user isolation and PreReplyDecision ownership. Treat 90% as a later objective that requires fresh evidence, not a promised outcome.

## Non-negotiable constraints

- Work only on branch `codex/longmemeval-goal-80`; preserve existing unrelated worktree changes.
- The local-Qwen judge is a diagnostic substitute following the official protocol, not an upstream GPT-4o score.
- No DeepSeek or other cloud model calls. All future model calls use the local Qwen endpoint and require frozen inputs.
- No Reader change without at least five independent direct artifact proofs of one general mechanism.
- PreReplyDecision remains the sole semantic authority. No benchmark IDs, question text, answer, labels, or hidden recall fallback may drive production behavior.
- Each behavior change needs non-benchmark positive and negative regression tests. Commit each independently validated batch.

## Phases

### Phase 0: Preserve the verified zero-Qwen attribution baseline
- [x] Inspect and validate the existing uncommitted Batch 0 analyzer/test/documentation changes.
- [x] Commit only those attribution artifacts separately if validation is still clean.
- **Status:** complete

### Phase 1: Correct d232ad8 semantic and isolation regression
- [x] Restore fail-closed structured-memory behavior for unresolved/ambiguous named subjects.
- [x] Add a narrow, observable timeline-only path for explicitly unresolved named-event evidence, supported by synthetic fixture evidence.
- [x] Split prior-user-event recall from prior-assistant-answer timeline recall in the PPD contract without fixed phrase matching.
- [x] Add synthetic positive, negative, ambiguity, and cross-user tests.
- [x] Correct the handoff/root-cause claims and arithmetic to verified facts only.
- [x] Commit the validated correction as a separate batch.
- **Status:** complete

### Phase 2: Frozen small local-Qwen replay
- [x] Build a frozen target/control manifest after Phase 1 passes zero-Qwen gates.
- [x] Run one small local-Qwen replay after inputs, cache behavior, and local-only environment are verified.
- [x] Decide from per-case evidence whether to continue this mechanism or roll it back.
- **Status:** complete

### Phase 3: Correct text-query versus continuous-audio routing
- [x] Ensure structural length alone cannot route ordinary text questions into continuous capture.
- [x] Preserve continuous-capture fast path for real audio events using existing audio provenance.
- [x] Add non-benchmark tests for long text, long audio, and semantic recall routing.
- [x] Run zero-Qwen regression gates and commit the isolated batch.
- [x] Run a frozen local-Qwen replay of all 11 original route-not-requested cases plus controls.
- [x] Review route, recall, and judge outcomes before choosing the next retrieval mechanism.
- **Status:** complete

### Phase 4: Evidence-led retrieval/state batch
- [x] Mine direct evidence across the remaining official-judge failures, including candidates, selected evidence, temporal ranges, and truncation boundaries.
- [x] Implement one general Timeline lexical-ranking mechanism with synthetic regressions.
- [x] Repeat a frozen small replay: targets 5/6 correct and controls 6/6 correct.
- [x] Mine and repair a PPD ISO-8601 temporal-boundary normalization defect with six direct frozen artifacts.
- [x] Build and run a frozen target/control replay for the ISO temporal-boundary repair; reject and revert it because targets remained 0/6.
- [x] Mine import/write-time provenance: 29 official failures contain 83 explicit English calendar records stored at conversation time with no parsed temporal text.
- [x] Reject the first parser-only trial because real import bypassed candidate-content parsing; revert it without stacking changes.
- [x] Repair the actual import contract: candidate date first, source chronology fallback, historical source time as the parser reference.
- [x] Rerun the six-case frozen import-provenance replay and verify original event rows carry the expected temporal metadata (`0/6 -> 6/6` targets; 17 persisted source-event date rows).
- [x] Extend the general explicit-date import replay to 21 direct source-event failures and six baseline-correct controls. Zero-Qwen write gate: `19/21` targets, 64 rows. Local-Qwen judge: targets `0/21 -> 10/21`; controls `6/6 -> 5/6`, with the sole miss a Reader execution failure rather than a semantic answer change.
- [ ] Mine an independent, source-grounded temporal-query contract that preserves a correct historical range when PPD emits a syntactically valid but semantically wrong ISO range; do not restore the rejected parser unchanged.
- **Status:** in_progress

### Phase 5: Reader/refusal evidence batch
- [x] Establish a directly evidenced Reader ledger-contract mechanism across five independent batch-ledger failures.
- [x] Gate source serialization and Reader-input reversibility before the Reader behavior change.
- [x] Run and independently judge one frozen local-Qwen replay for the included-item/source-decision contract repair.
- [x] Reject Phase 9 as a before/after decision gate for the current goal: the current 379/500 baseline and the replay both use source snapshot `24da24d`, while its target/control IDs were selected from an older baseline. It therefore cannot measure this repair's effect on the current 121 failures.
- **Status:** complete (historical evidence preserved; no score attribution retained)

### Phase 6: Re-anchor investigation to the current 379/500 baseline
- [x] Produce a new zero-Qwen attribution artifact selecting exactly the current baseline's 121 official-judge failures, with source hashes and per-case evidence paths.
- [x] Separate direct route/retrieval/write/Reader evidence from insufficient evidence without reusing older failure buckets as conclusions.
- [x] Add evaluator-only temporal-filter observability for the 11 current retrieval-empty rows; it distinguishes applied record-time filters from unresolved temporal input without Oracle or runtime product imports.
- [x] Establish the complete-set Reader source reversibility gate for 29 current judge-failed but valid-ledger rows: every declared source has full-text hash provenance, multiline coverage, and no dangling ledger source ID.
- [x] Repair the deterministic ledger-to-final-answer serialization boundary so temporal obligations retain source-backed occurrence/record timestamps; preserve source scope and do not infer an event time from record time.
- [x] Define and hash fresh current-baseline target, health-control, and refusal-control sets for the temporal-evidence serialization mechanism; verify all 17 IDs are unique and present in the fixed dataset without model calls.
- [x] Run the fixed-input local-Qwen replay: all 17 IDs completed with zero runner errors, but only 4/8 predeclared targets reached the valid-ledger temporal-evidence path.
- [x] Stop this mechanism before judge and revert its unverified Reader behavior; a post-run subset cannot replace the frozen target cohort.
- [ ] Add evaluator-only validated source-decision telemetry for complete-set outcomes, then use it only to diagnose a future fixed-input selection cohort; do not change selection or Reader behavior without a five-case direct mechanism.
- [x] Add validated source-decision telemetry and run the frozen Phase 11 telemetry-only cohort: 30/30 completed, zero runner errors, and no judge.
- [x] Stop the complete-set selection path without a product change: 15/19 target ledgers were valid, but eight changed their computed value versus the fixed baseline while the code changed only debug telemetry; no stable five-case source-decision defect is established.
- **Status:** in_progress

### Phase 12: PPD temporal compatibility gate
- [x] Audit the current-baseline PPD temporal payload population before treating PPD normalization as a product improvement.
- [x] Prove the classifier lacks the runtime reference clock required to resolve relative ranges.
- [x] Preserve direct execution only for a valid normalized PPD range; restore the existing authorized range resolver for incomplete PPD payloads.
- [x] Add non-benchmark valid-range and incomplete-range regressions; run zero-Qwen focused checks.
- [ ] Resume direct-evidence mining of the current 121 judge failures. This compatibility gate is not a score-improvement candidate and does not authorize a Qwen replay.
- **Status:** complete

### Phase 13: Reader execution-failure observability
- [x] Freeze every current-baseline official-judge wrong `execution_failed` case with health and refusal controls.
- [x] Check raw immutable per-case artifacts before spending a Qwen replay; the detail summary had omitted existing `execution_attempts` and validation errors.
- [x] Add a standard-library evaluator-only audit so future full runs retain this exact error-signature analysis without manual case inspection.
- [x] Reject a Reader behavior patch: the largest exact error signature is four cases, below the five-case mechanism gate.
- [x] Complete the frozen observability replay: all 20 exact manifest questions completed, zero runner errors, one worker, disabled cache, fixed dataset hash, and no judge.
- [x] Compare raw Reader outcomes with the current baseline. The fresh import produced changed Reader contexts for every target and health control, so its output transitions cannot isolate a Reader mechanism; health and refusal controls also regress.
- [x] Reject a Reader behavior patch and a local-Qwen judge for this replay; preserve the artifact as non-score diagnostics only.
- **Status:** complete (diagnostic only; Qwen observability run, no judge and no product change)

### Phase 14: Frozen Reader-input replay capability
- [x] Design an evaluator-only input freezer for later Reader-only replay from immutable per-case `result.json` context. It does not import history, invoke PPD, call a Reader, or use a judge.
- [x] Make it write a new immutable input manifest containing case identity and hashes for question, answer task, recall context, and source envelopes before any later model call.
- [x] Add fixture tests for duplicate/missing IDs, source-hash mismatch, output-directory collision, source-run immutability, and no product/Reader/model imports.
- [x] Freeze the first all-shape cohort: 29 targets, 46 health controls, and 20 refusal controls; validate all 95 input hashes with zero model/judge/network calls.
- [x] Replace the v1 approximate task summary with the exact shared Reader task builder used by the Runner, freezer, and replay executor.
- [x] Seal v2 configuration, Reader/code snapshots, full non-secret invocation payload hashes, and all 95 source artifacts before a Reader can be constructed.
- [x] Keep v1 as an immutable historical artifact but mark it non-executable; freeze v2 separately with `qwen_calls=judge_calls=network_calls=0`.
- [x] Do not run local Qwen. The replay executor is tested only with a fake Reader and has no judge entrypoint.
- **Status:** complete (capability sealed; no Reader replay authorized or executed)

### Pause: Android work window
- [x] Preserve PPD temporal compatibility logic only on `codex/longmemeval-goal-80`; do not migrate it to `dev_ykhu` or `main`.
- [x] Record the comparable 2026-09-03 run as `370/500`, not an improvement over the 2026-09-02 `379/500` baseline; do not infer PPD causality.
- [x] Stop LongMemEval work after Phase 14 closeout. Do not run Qwen, judge, or a full 500-case evaluation during the Android work window.
- [ ] On resume, first re-audit PPD separately and mine one source-traceable general mechanism across at least five cases before any small replay.
- **Status:** paused (80% goal is neither complete nor blocked)

## Acceptance

- 80% is reached only by a fresh comparable 500-case local-Qwen run, with source/input hashes and per-type deltas retained.
- Strong categories must not silently regress; every full replay reports category deltas and abstention controls.
- Every commit has focused tests, compile and diff checks; any model replay is separately recorded.

## Errors encountered

| Error | Resolution |
| --- | --- |
| `git switch -c codex/longmemeval-goal-80` could not create `.git/refs` under the workspace sandbox. | Repeated once with user-authorized Git metadata access; branch created successfully. |
| `.planning/.active_plan` is an existing symlink to an old plan and could not be overwritten. | Leave it intact; use this goal directory explicitly and do not delete/replace the old symlink without separate authorization. |
| One broad documentation patch did not apply because a target heading differed from the reviewed snapshot. | Do not retry the broad patch; update each affected claim in small, file-specific patches after re-reading its current text. |
