# LongMemEval Local Data

Place the downloaded LongMemEval JSON files in this directory:

- `longmemeval_oracle.json`
- `longmemeval_s_cleaned.json`

These files are local benchmark inputs and are ignored by Git.

The Oracle runner creates a timestamped directory under `reports/longmemeval/`.
Its primary artifacts are:

- `longmemeval_oracle_memory.jsonl`: evaluator input containing only
  `question_id` and `hypothesis`.
- `longmemeval_oracle_memory.jsonl.details.json`: per-question import, recall,
  error, and pre-truncation `recall_context_chars` details. Fragment failures
  include their source session and turn diagnostic; those items do not proceed
  to the Reader with incomplete history.

The default import mode replays each original session through the production
conversation-import API. User-authored facts pass through the normal memory
gate; assistant replies remain linked Timeline evidence and are not stored as
standalone personal memories. Oracle `answer`, `has_answer`, and answer-session
metadata are never passed into import, recall, or the reader.

Use `--limit 2` for a paid smoke run and `--limit 0` only when intentionally
running all matching questions. Explicit `--question-id` selections are never
truncated by the default limit. The runner shows live progress on stderr.
