# Findings: LongMemEval Goal

## 2026-09-03: Current 379/500 baseline re-attribution

- Immutable artifact: `reports/longmemeval/pref500-20260902-182616-35190-attribution-20260903/`.
- The artifact validates exactly 500 detail IDs, 500 official-judge IDs, 121 current judge failures, and case artifacts for all 121. It records `qwen_calls=0`, `judge_calls=0`, and `network_calls=0`.
- Direct evidence identifies 11 `retrieval_empty` and 3 `route_not_requested` rows. The remaining 107 rows are deliberately `insufficient_artifact_evidence`; they are not proof of a Reader, retrieval, or write defect.
- Wrong-case distribution: multi-session 41, temporal-reasoning 37, single-session-preference 18, knowledge-update 11, single-session-assistant 10, single-session-user 4. Behavior overlays remain independent: 33 should-answer-but-refused, 9 should-abstain-but-answered, and 79 other judge mismatches.
- The next investigation must start with the 11 current `retrieval_empty` cases and prove a shared current-code mechanism before any product change.

## 2026-09-03: Retrieval-empty cohort triage (no product change)

- Seven of the 11 current `retrieval_empty` rows carry one observable signature: PPD requests recall and marks a temporal expression, but supplies no usable `start_at`/`end_at`; final temporal state is `pre_reply_decision_invalid_range` with reason `structured_temporal_query_missing_valid_range`, then Reader receives no context and refuses.
- The seven include natural relative/date/duration language and do not all expose the same PPD `temporal_text`, so this is a candidate for source-grounded temporal resolution, not evidence for a phrase-specific rule. The other four retrieval-empty rows lack this signature and must not be folded into it.
- A previous ISO-only PPD-bound normalization trial was rejected on its frozen replay. Any new investigation must preserve semantic temporal resolution rather than treat a valid-looking serialized range as automatically correct.
- Current code confirms that an invalid PPD temporal range does not itself end the path: when event recall is authorized and the planner range is unusable, `GlassesChatService.chat()` calls the existing isolated temporal resolver on the user message. The next audit must inspect that runtime resolver output; changing PPD range normalization alone would address a symptom, not yet a proven root cause.
- Runtime temporal outputs split the seven cases. Some produce a usable range that appears to constrain memory occurrence time even though the date-like phrase qualifies the requested answer (for example, a historical object attribute or recency of recommended material); others produce an uncertain/incorrect event-time range; others correctly decline to invent an anchor for an event comparison. No five-case, one-mechanism product repair is proved yet.
- The next safe batch is evaluator-only observability: record whether a temporal range was applied as a memory-record filter, its scope, candidate/selected counts, and whether the date phrase operates on the requested answer versus the evidence record. It must not use Oracle answers or affect runtime behavior.

## 2026-09-03: Temporal-filter audit completed

- Immutable artifact: `reports/longmemeval/pref500-20260902-182616-35190-temporal-filter-audit-20260903/`. It consumes the current zero-Qwen attribution output and the frozen report details only; 500 detail IDs and exactly 11 retrieval-empty rows were validated with zero model, judge, and network calls.
- Of the 11 rows, five runtime temporal resolutions are usable, but only three actually apply a record-time filter: two `temporal_range` flows and one complete-set scope. Two usable temporal resolutions flow through text search without record-time filtering. The remaining six have no usable range or have explicit skipped recall.
- Therefore temporal filtering cannot explain the whole retrieval-empty cohort, and the three applied-filter rows do not yet meet the five-case product-change threshold. Keep every row marked `manual_review_required`; no PPD, Reader, or retrieval behavior change is authorized by this audit.

## 2026-09-03: Answerable-refusal cohort triage (no product change)

- Of 33 current should-answer-but-refused rows, 14 fail at empty context, 17 at Reader evidence, and two at coverage. The 17 evidence-stage rows have nonempty Reader context (563 to 12,503 characters) and all record an empty selected_source_ids list; 15 still report one or more relevant_evidence items.
- This is a candidate Reader evidence-binding mechanism, but not yet a root cause: a comparison against current judge-correct answered cases is required to establish whether empty selected_source_ids is abnormal rather than ordinary debug shape. Do not modify Reader behavior yet.
- Comparison rejected that candidate: 314 current judge-correct answered cases also have nonempty context and empty selected_source_ids, so that field is ordinary debug shape rather than evidence-binding failure. Do not change Reader source selection or serialization from this signal.
- Nine of the 17 answerable evidence-stage refusals carry the route-owned uncertainty policy state_limits_when_context_is_sparse, while eight explicitly allow abstention. The nine meet the count threshold for contract investigation, but not yet for a Reader repair: the ordinary AnswerDirective prompt is a main-service instruction path, whereas LongMemEval uses a dedicated Reader path that must be inspected separately.
- Dedicated Reader inspection rejects a prompt-propagation repair: it already receives the answer task, explicitly forbids unknown answers when it listed relevant evidence, and automatically retries once after a refusal. All nine state-limits rows exhausted that retry. Their reported snippets include background-only evidence in some cases, so forcing an answer would risk unsupported claims. Stop this Reader path rather than stack another prompt rule.

## 2026-09-03: Current coverage audit

- Immutable artifact: reports/longmemeval/pref500-20260902-182616-35190-coverage-audit-20260903/. It is evaluator-only and uses the benchmark answer reference only as offline diagnostic comparison; it never affects runtime and records zero Qwen, judge, and network calls.
- Ten current failures have direct stored-source evidence absent from Reader context. They split into three candidate-dropped-before-context, three stored-source-not-in-candidate-trace, and four timeline-trace-missing rows. This split is below the five-case threshold for any ranking, Top-K, or Reader change.
- Route inspection of those ten preserves the split: three timeline-trace-missing rows request raw evidence while not requesting timeline recall, one has no recall route, three use timeline supplements but lose the supporting source before context, and the remaining cases have distinct temporal or candidate paths. No hidden fallback or route rule can be justified from this group.

## 2026-09-03: Whole-failure topology and Reader output-shape gate

- Among all 121 current judge failures, 73 are answered with nonempty context, 17 refuse at evidence, 14 refuse at empty context, five fail `batch_ledger`, and four fail `final_answer`. This is a topology summary, not a causal attribution.
- The five batch-ledger failures do not prove one new Reader defect: three violate `included_source_without_item`, one produces a nonincluded item without source IDs, and one mixes included units. All are complete-set inputs, but 83 other complete-set inputs do not fail the batch ledger. Do not add a second ledger prompt or relax validation.
- Twenty-nine judge-wrong answered cases have context but no `relevant_evidence`; 46 judge-correct answered cases share that same debug shape. The existing parser accepts final-answer-only JSON, so the stored debug cannot tell omitted field from an explicit empty list. This correlation is insufficient for a behavior change.
- Added evaluator-only output-schema observability: `evidence_and_final_answer`, `final_answer_only`, or `unstructured_text`. It records only a shape label, not model raw text, and preserves the exact answer/retry path. A future frozen replay can now tell whether omitted evidence fields correlate with failures before any Reader behavior proposal.
- Corrected a measurement bug found while checking apparent truncation: result-level `reader_input_chars` always used the ordinary single-prompt truncation formula. Complete-set Reader calls instead batch structured sources and may consume the full context. The result field now prefers the Reader's actual debug input length; this changes reporting only, not Reader input, limits, or source delivery.
- Existing current-baseline case artifacts preserve final answer text and summarized Reader debug, but not the raw structured Reader response. Consequently the 29 `relevant_evidence=[]` answered rows cannot be retrospectively separated into an omitted field versus an explicit empty selection. The new output-shape label only applies to future fixed-input runs. Do not spend a Qwen replay merely to fill this historical observability gap.

## 2026-09-03: Complete-set temporal evidence serialization repair

- The immutable complete-set audit selected 29 current official-judge failures with `ledger_valid=true`. Every row is reversible: declared source ID equals inventory source ID, text is nonempty and hashed, multiline status is retained, accounted source count matches, and ledger item source IDs are a subset of the inventory. Source delivery and source serialization into the ledger are therefore not the shared defect.
- Nine selected rows are temporal-reasoning. Source candidates arrive at the ledger stage with `occurred_at` or `recorded_at`, while `validate_aggregation_ledger()` removes `_time` from the validated items. The final-answer prompt then receives no deterministic time field even when the fixed obligation is `temporal_relation`. This is directly evidenced at least five times and is independent of question text or IDs.
- The repair enriches only the final validated ledger payload with `temporal_evidence` reconstructed from the already validated item source IDs. It preserves both timestamps, labels record time as non-event evidence, and never serializes an unselected or untimestamped source. It does not alter the strict ledger validator, PPD, recall, source count, or aggregation value. Synthetic positive, missing-time negative, and unselected-source isolation tests pass together with source-envelope and evidence-set regressions (`86 passed`); no Qwen call occurred.

## 2026-09-03: Phase 10 temporal handoff replay rejected before judge

- The frozen `b26a3ef` local-Qwen replay completed all 17 exact IDs at `reports/longmemeval/pref500-phase10-20260903-b26a3ef/`, with one worker, disabled cache, the fixed dataset hash, and zero runner errors.
- Its required pre-judge mechanism gate failed: four of eight fixed target cases reached `ledger_valid=true` and carried temporal evidence (3/7/6/9 source counts respectively); three targets answered via a nonvalid-ledger fallback and one failed `batch_ledger`. The source-scoped handoff cannot explain their answers in this run.
- Three baseline-correct health controls remained answered and all six baseline-correct refusal controls remained `insufficient_evidence`, but this does not rescue attribution because the frozen target cohort was only half exposed to the change.
- No local-Qwen judge was run. Post-hoc judging only the four exposed targets would be selection after observing the run and is forbidden. Revert the Reader handoff rather than stack a ledger reliability or prompt change on this signal; retain the audit/replay artifacts and investigate stable complete-set execution separately.

## 2026-09-03: PPD temporal contract needs a compatible executor path

- The 379/500 baseline stores 176 PPD decisions with `has_expression=true`; all 176 omit executable start/end bounds. They include 82 substitute-judge-correct results (27 temporal-reasoning), so converting every incomplete payload directly into an abstention has a proven regression surface.
- `classify_pre_reply_decision()` accepts the message, recent capsule, discussion catalog, and policy metadata, but no runtime reference clock or evaluator question date. It therefore cannot consistently derive absolute timestamps for relative text such as `last week`, nor for memory-relative anchors such as `after the first service`.
- Valid normalized PPD ranges remain direct executor inputs. For incomplete PPD ranges, restoring `resolve_temporal_expression()` preserves the PPD decision that time recall is needed while supplying the existing runtime reference clock. This is not a second planner or a change in recall authorization.

## 2026-09-03: Reader execution failures are not one repairable mechanism

- The current detail JSON had only status and stage, but each immutable case `result.json` retains `execution_attempts`, `ledger_error`, and `ledger_validation_errors`. This required no rerun, product import, Qwen, judge, or Oracle answer.
- The nine current judge-wrong execution failures split exactly into four `final_answer_missing_verified_value`, three `included_source_without_item`, one nonincluded-item/source-ID violation, and one mixed-unit violation. They cross batch-ledger and final-answer stages.
- Four is below the five-case change gate. The errors are emitted by the strict evidence contract, so a prompt/validator/Reader patch would be an unsupported attempt to merge heterogeneous failures.

## 2026-09-04: Phase 13 reader observability replay cannot isolate a repair

- `reports/longmemeval/pref500-phase13-20260903-071b8c8/` is terminal: all 20 manifest IDs completed, zero runner errors, fixed dataset SHA-256, one worker, disabled cache, and no judge was run.
- The nine original execution-failure targets transition to six answered and three execution-failed outputs. That is not a repair result: all 9 target contexts and all 6 health-control contexts have changed bytes relative to the current 500 baseline after a fresh import.
- The controls explicitly reject a favorable interpretation: one of six baseline-correct health cases becomes execution-failed and one of five baseline-correct refusal cases produces an unsupported answer. There was no Reader product change in this batch.
- Preserve the run as diagnostics, do not judge it, and do not derive a Reader/validator/prompt modification. Return to current-baseline direct-evidence mining.

## 2026-09-04: Fixed Reader inputs are now independently reproducible

- The current baseline did not retain raw structured Reader output, and the fresh Phase 13 import changed all target and health Reader contexts. A fresh import cannot isolate Reader behavior.
- `longmemeval_reader_input_manifest` therefore freezes hashes for the real question, derived answer-task contract, full recall context, and recalled source envelopes from an already completed run. It is standard-library-only and has no product service, Reader, judge, model, or network import/call.
- The first immutable cohort contains every current wrong answered/nonempty/empty-retained-evidence row (29), every matching current correct row (46), and every current correct nonempty-context refusal control (20). Its 95 IDs are selected by artifact state rather than question text, answer, label, or ID pattern.
- The output `pref500-20260902-182616-35190-reader-input-manifest-20260904` has unique IDs and all required hashes with zero calls. It prepares, but does not authorize, a fixed-input Reader replay; the later executor must revalidate each hash before Qwen is allowed.

## 2026-09-04: Phase 14 v1 sealed as history; v2 is the only executable input contract

- The initial v1 manifest is not safe for a fixed Reader replay: it hashes an approximate debug-task shape, while the normal Runner normalizes obligations, omits legacy no-op tasks, and conditionally carries complete-set fields. Its draft executor also accepted mutable temperature, token, context-length, timeout, and thinking flags. Do not run v1.
- The pure eval-side `extract_answer_task_from_debug()` now supplies the same normalized task to the normal Runner, input freezer, and fixed replay executor. It changes no product route, PPD decision, recall result, Reader prompt, validator, or answer behavior.
- The v2 manifest hashes every actual non-secret Reader argument: question, type, date, exact answer task, full recalled context, recalled memory/timeline inputs, and frozen Reader configuration. It also pins both the Reader implementation and task-builder source hashes. The executor checks all of those plus every source result and completed marker before Reader construction.
- The 2026-09-02 run manifest did not persist timeout or thinking mode. V2 records them explicitly as the then-current Runner defaults and labels their provenance `historical runner default`; all other Reader settings originate directly in the source run manifest. This is an explicit historical assumption, not an unrecorded claim.
- New immutable v2 artifact: `reports/longmemeval/pref500-20260902-182616-35190-reader-input-manifest-v2-20260904/`, exactly 95 cases (29 target / 46 health / 20 refusal), zero Qwen, judge, and network calls. It is only a replay capability, not a score or a Reader behavior result.

## 2026-09-04: Goal paused for Android work

- Comparable full-run evidence is unfavorable but not causal: `pref500-20260903-183545-29620` finished 500/500 with zero runner errors and a local-Qwen substitute-judge result of `370/500` (74.0%), nine below the 2026-09-02 `379/500` baseline. The source snapshot is `071b8c8`, not the later evaluator-only `d461748` commit.
- Keep PPD temporal compatibility code on the goal branch only. Do not migrate it to `dev_ykhu` or `main`, and do not interpret the single stochastic full run as a PPD verdict.
- Resume only after a fresh direct-evidence selection identifies one general, source-traceable memory-kernel mechanism in at least five cases. No automatic Qwen replay, judge, or 500-case run is authorized while Android work is active.

## 2026-09-03: Preference obligation-completion hypothesis rejected

- All 18 current preference failures use the generic `personalized_recommendation` task. Fourteen Reader calls answered and four returned `insufficient_evidence`; this rules out a missing task route but does not prove an answer-generation mechanism.
- The 12 current judge-correct preference cases self-report all fixed obligations in `coverage`. Five failure rows do not, but three are refused or empty-context paths. Only two rows across all 121 failures have all of: an answered final response, nonempty context, at least one listed relevant evidence snippet, and a missing fixed obligation.
- The coverage list is a model self-report without retained source IDs for best-evidence Reader calls. Two cases cannot meet the required five-case, full-source/reversibility gate. Do not add prompt wording, a retry, or a coverage validator; this result is a rejected hypothesis, not a repair.

## 2026-09-03: Phase 11 complete-set decision telemetry stops the selection path

- The frozen telemetry-only run completed 30/30 at `reports/longmemeval/pref500-phase11-20260903-57b1751/`, zero runner errors, no cache, one worker, local Qwen only, and no judge. It used the debug-only telemetry commit `57b1751`.
- Fifteen of 19 valid-ledger multi-session target replays supplied source decisions. Across those target ledgers, the validated decisions contain 106 `included`, 410 `excluded`, and six `uncertain` dispositions. Each remains bound to the strict source-decision/item validator; the telemetry does not create or alter any decision.
- Eight of the 15 valid target ledgers compute a different total from the current baseline under the same data and Reader settings. The run contains both direction changes (for example baseline `3` to replay `2`, `2` to `3`, and `0` to `15`), so it does not expose a stable common source-selection error. Four target replays never reached a valid ledger, which reinforces rather than resolves the stability issue.
- This batch is diagnostic only. Do not judge it, post-select favorable values, or add a source-selection/Reader patch. The next mechanism must come from a different direct artifact class, with source hashes and a reproducible five-case code path.

- Baseline: `pref500-20260831-182840-17656`, 354/500 = 70.8% using the official-protocol local-Qwen substitute judge.
- Batch 0 attribution is complete and immutable: 12 `retrieval_empty`, 11 `route_not_requested`, 123 `insufficient_artifact_evidence`. Reader refusal is an overlay, not a Reader root cause.
- Commit `d232ad8` changed unresolved named-subject recall to fall back to self. Focused tests currently prove this breaks two established fail-closed boundaries; it must be corrected before replay.
- Of the 11 `route_not_requested` cases, only five explicitly request prior assistant conversation content. The other six are count, cost, temporal, or knowledge-update questions and require separate attribution.
- `d232ad8` could theoretically affect 13 cases (11 + 2), giving 367/500 = 73.4% if every one flipped. It cannot justify the documented 75.4% claim, which assumes all 23 verified cases are fixed.
- The corrected mechanism does not use any benchmark question or phrase list: unresolved named subjects keep empty structured-memory scope; only same-user raw Timeline is scanned for complete-set recall when the name is unambiguous. Ambiguous names remain fail-closed.
- The classifier contract now distinguishes evidence source generically: prior assistant content uses raw Timeline; user actions/events use personal event recall. It does not infer the route from trigger phrases.
- Frozen 13-case local-Qwen replay at `f8c354f` completed with all cases successful. Local-Qwen judge: controls 6/6 remained correct; target cases improved 0/7 to 3/7, for 9/13 overall. Both unresolved-named targets flipped to correct; one of five prior-assistant targets flipped.
- Ten of eleven original `route_not_requested` cases share a direct mechanism: `plan_turn` applies `continuous_capture` from text length and punctuation alone, so PPD is never called. This affects ordinary text LongMemEval queries as well as intended continuous audio input.
- Frozen Phase 3 replay at `42c8391` completed 19/19 cases with cache disabled, workers=1, and local Qwen only. Completed IDs exactly match the frozen manifest. The local-Qwen judge result is 15/19: all 6/6 controls held; 8/11 former route failures flipped correct; one of two unresolved-name regression targets flipped correct.
- All 11 former route failures now contain `pre_reply_decision_applied=true`, proving the causal routing fix reached the intended decision layer. This is a small diagnostic replay, not a projected or formal 500-case score.
- Four replay failures divide into four different post-routing observations: empty recalled context (`0e5e2d1a`), recalled-but-unselected evidence (`c7cf7dfd`), temporal ordering not resolved from present evidence (`gpt4_18c2b244`), and a named-event temporal ordering error (`gpt4_4929293a`). They are not enough direct evidence for a Reader or retrieval patch.
- The new immutable zero-Qwen coverage audit finds 13 wrong cases whose direct source reference is stored in the per-case export but absent from Reader context. Six different cases prove the supporting Timeline source was returned to the lexical candidate pool then placed in `dropped_candidate_ids` at the five-chunk boundary. The audit excludes the final question turn to avoid treating answer words echoed by the question as support.
- These six direct cases cross four question types and share a product mechanism: Timeline FTS and fallback terms still score English stopwords, whereas structured-memory search already removes them. The next repair is limited to parity of Timeline query-term filtering and its synthetic retrieval tests; it is not an increase to Top-K or a Reader change.
- Commit `bf1b30a` applied that narrow parity change. On frozen SQLite exports, the direct source re-entered the selected five Timeline chunks for 5/6 target cases. Local-Qwen small replay then scored targets 5/6 and controls 6/6, all 12/12 completed, with no Reader execution failure. One target stayed wrong even with direct answer terms visible, so it is not evidence for another ranking increase.
