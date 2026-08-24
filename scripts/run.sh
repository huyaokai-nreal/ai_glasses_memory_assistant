# 默认走缓存；若想强制重测 import/recall（测性能），再加 LONGMEM_NO_CACHE=1。
# 保留 runner 的实时 stderr 进度条，不让外层缓冲输出。
export PYTHONUNBUFFERED=1
# 这是全新 500 题跑法而非 --resume；默认每次创建独立目录，避免旧 checkpoint 冲突。
EVAL_OUT="${LONGMEM_OUT:-reports/longmemeval/pref500-$(date +%Y%m%d-%H%M%S)-$$}"
EVAL_WORKERS="${LONGMEM_WORKERS:-1}"
if [[ -e "$EVAL_OUT" ]]; then
  echo "ERROR: fresh-run output already exists: $EVAL_OUT" >&2
  echo "Choose a new LONGMEM_OUT, or use the dedicated resume script for an existing run." >&2
  exit 1
fi
exec env LONGMEM_DIRECT_HOST=10.252.17.5 LONGMEM_NO_CACHE=1 LONGMEM_WORKERS="$EVAL_WORKERS" \
  LONGMEM_OUT="$EVAL_OUT" \
  bash scripts/run_longmemeval_server_llm.sh 0 import
