#!/usr/bin/env bash
# Run LongMemEval through a LOCAL LLM (qwen3.8-27B via llama.cpp llama-server) on a
# remote GPU box, reached from this Mac directly (LONGMEM_DIRECT_HOST) or via SSH tunnel.
# Zero cloud API cost. 27B is the only test LLM AND judge.
#
# Why this exists:
#   Testing the memory system (recall + structured-memory write + answer) should
#   not cost cloud-LLM money, and the on-device product may not even use DeepSeek.
#   This script points the ENTIRE pipeline at a local Ollama model:
#     - the import phase (which WRITES structured memories) uses the agent's LLM
#     - the reader phase (answer generation) uses --reader-*
#     - the judge phase (official scoring) uses the env LLM
#   All three read AI_GLASSES_LLM_* , so setting those env vars covers everything.
#
# No repo code is modified. Only this new script is added.
#
# Usage:
#   bash scripts/run_longmemeval_server_llm.sh [LIMIT] [HISTORY_MODE]
#   LIMIT        default 30 (use 0 for all 500).   e.g. 30 to validate, 0 for full
#   HISTORY_MODE default import  (import = writes structured memories from oracle
#                conversations, then recalls; matches pref500-20260812-e31d5c)
#   By default this runs the 30 single-session-preference questions, matching the
#   existing DeepSeek baseline reports/longmemeval/pref30_20260813 for easy compare.
#
# Env overrides:
#   LONGMEM_SERVER  ykhu@10.252.17.5   (GPU box running ollama)
#   LONGMEM_MODEL   qwen3.8-27b-32k    (default; 27B is the ONLY test LLM + judge now)
#   LONGMEM_ORACLE  data/benchmarks/longmemeval/longmemeval_oracle.json
#   LONGMEM_QTYPE   single-session-preference  (question type to filter; "" or "all" = run all categories)
#   LONGMEM_NO_CACHE  any value  (OPTIONAL: bypass the L2 per-question cache. The cache
#                    key includes the app model + source hash, so switching models or
#                    changing import/recall code auto-invalidates it. Set this only to
#                    force a fresh import+recall, e.g. when measuring import performance)
#   LONGMEM_WORKERS 8  (parallel question workers; with --no-cache each question imports
#                    its own memory library independently, so raising this speeds the run up
#                    a lot while the GPU has headroom. Real ceiling is the llama-server's
#                    n_parallel slots — beyond that requests just queue; watch for GPU idle)
#   LONGMEM_OUT     reports/longmemeval/local-<date>-qwen32k
#   LONGMEM_SERVER_PORT 11438        (remote 27B llama-server port; 14B/ollama removed)
#   LONGMEM_DIRECT_HOST ""           (if set, e.g. 10.252.17.5, skip the SSH tunnel and
#                                    hit http://HOST:SERVER_PORT/v1 directly; needed when the
#                                    local sandbox/proxy breaks large requests over the tunnel)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Help: `bash scripts/run_longmemeval_server_llm.sh --help` (or -h)
# Intercept BEFORE positional parsing so --help is not swallowed as LIMIT.
# ---------------------------------------------------------------------------
print_help() {
cat <<'EOF'
================================================================================
 run_longmemeval_server_llm.sh  —  LongMemEval 本地 27B 评测启动器
================================================================================
用 A100 服务器上的 qwen3.8-27B（llama-server :11438）跑完整的
longmemeval oracle 500 题评测。整条链路（import 写记忆 / reader 答题 /
judge 打分）都走这一个模型：它既是测试 LLM，也是 judge。无云端 API 费用。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
  bash scripts/run_longmemeval_server_llm.sh [LIMIT] [HISTORY_MODE]
  bash scripts/run_longmemeval_server_llm.sh --help | -h

位置参数:
  LIMIT        0 = 跑全部 500 题；30 = 前 30 题（快速验证管线）；N = 前 N 题
               （默认 30）
  HISTORY_MODE import（默认，先写结构化记忆再召回，最贴近产品真实管线）/
               timeline / chat

--------------------------------------------------------------------------------
环境变量（写在命令前面覆盖默认行为）
--------------------------------------------------------------------------------
  LONGMEM_MODEL        qwen3.8-27b-32k   唯一测试 LLM + judge（一般不动）
  LONGMEM_DIRECT_HOST  不设               设 10.252.17.5 直连 GPU 盒、跳过 SSH
                                          隧道；大请求防 502/404（推荐直连）
  LONGMEM_SERVER       ykhu@10.252.17.5   仅隧道模式用的 GPU 盒地址
  LONGMEM_SERVER_PORT  11438             远端 llama-server 端口（14B 已移除）
  LONGMEM_PORT         11435             仅隧道模式的本地映射端口
  LONGMEM_QTYPE        ""（全 6 类）       只跑某一类: multi-session /
                                          temporal-reasoning / knowledge-update /
                                          single-session-user / -assistant / -preference
  LONGMEM_NO_CACHE     不设               （可选）设任意值(如 1) 绕过 L2 缓存，强制重跑
                                          import+recall；缓存 key 已含 model+代码 hash，
                                          换模型/改代码会自动失效，一般无需设
  LONGMEM_WORKERS      8                  并行题数；no-cache 下每题独立建库，
                                          调高大幅提速；天花板=服务端 slots
  LONGMEM_OUT          reports/longmemeval/local-<日期>-qwen32k
                                          ★输出目录，每次重启用新目录
  LONGMEM_ORACLE       data/benchmarks/longmemeval/longmemeval_oracle.json
                                          题库路径（一般不动）

--------------------------------------------------------------------------------
常见场景（复制即用）
--------------------------------------------------------------------------------
1) 标准全量 500（直连 + no-cache + 8 路并行）:
   LONGMEM_DIRECT_HOST=10.252.17.5 LONGMEM_NO_CACHE=1 LONGMEM_WORKERS=8 \
   LONGMEM_OUT=reports/longmemeval/local-20260819-qwen27b-full500-nocache-v2 \
   bash scripts/run_longmemeval_server_llm.sh 0 import

2) 小批量验证管线（30 题，最快）:
   LONGMEM_DIRECT_HOST=10.252.17.5 LONGMEM_NO_CACHE=1 \
   LONGMEM_OUT=reports/longmemeval/local-20260819-smoke \
   bash scripts/run_longmemeval_server_llm.sh 30 import

3) 只跑某一类（如 multi-session，定位短板）:
   LONGMEM_DIRECT_HOST=10.252.17.5 LONGMEM_NO_CACHE=1 LONGMEM_QTYPE=multi-session \
   LONGMEM_OUT=reports/longmemeval/local-20260819-multi-session \
   bash scripts/run_longmemeval_server_llm.sh 0 import

4) 隧道模式（无直连、走 SSH，仅当直连 IP 不通时）:
   LONGMEM_NO_CACHE=1 LONGMEM_OUT=reports/longmemeval/local-20260819-tunnel \
   bash scripts/run_longmemeval_server_llm.sh 0 import

--------------------------------------------------------------------------------
三个必记的坑
--------------------------------------------------------------------------------
  • LONGMEM_OUT 必须每次换新目录——脚本不带 --overwrite，目录已存在会直接报错退出。
  • 缓存 key 已含 app model + 源码 hash：换模型或改 import/recall 代码会自动失效重建，
    不再需要 LONGMEM_NO_CACHE=1；它仅用于强制 fresh 重跑（如测 import 性能）。
  • 后台跑用 nohup ... > log 2>&1 &，进度在日志里；看进度条另开终端跑
    bash scripts/longmemeval_progress.sh [OUT_DIR] [TOTAL]。

跑前请确认服务端存活: curl http://10.252.17.5:11438/v1/models 应返回 qwen3.8-27b-32k
================================================================================
EOF
}
case "${1:-}" in
  -h|--help|help)
    print_help
    exit 0
    ;;
esac

SERVER="${LONGMEM_SERVER:-ykhu@10.252.17.5}"
# qwen2.5:14b-32k is a server-side Modelfile variant pinned to num_ctx 32768.
# The plain qwen2.5:14b gets reloaded + truncated to 2048 when request contexts
# vary, which both slows it (30-72s reloads) and corrupts long-history imports.
# 32768 covers the longest single session in LongMemEval (~17.5k tokens) with headroom.
MODEL="${LONGMEM_MODEL:-qwen3.8-27b-32k}"
LIMIT="${1:-30}"
HISTORY_MODE="${2:-import}"
# Parallel question workers. Each question's import library is isolated per-question,
# so raising this is safe and speeds up a --no-cache run substantially.
WORKERS="${LONGMEM_WORKERS:-8}"
# Empty or "all" => no question-type filter (run all 500 categories).
# NOTE: use ${VAR-...} (no colon) so an explicit empty LONGMEM_QTYPE="" is preserved
# as empty and skips --question-type. The old ${VAR:-...} collapsed "" to the default
# and silently filtered to single-session-preference only (30 questions, not 500).
# Default is now "" (all categories) since the full 500 is the standard run.
QTYPE="${LONGMEM_QTYPE-}"
if [ "${QTYPE}" = "all" ]; then QTYPE=""; fi
ORACLE="${LONGMEM_ORACLE:-data/benchmarks/longmemeval/longmemeval_oracle.json}"
OUT="${LONGMEM_OUT:-reports/longmemeval/local-$(date +%Y%m%d)-qwen32k}"
# NOTE: the Mac itself already runs a local ollama on 11434, so the tunnel to the
# GPU box uses 11435 locally to avoid the port clash. 11435 -> SERVER:SERVER_PORT.
# 14B runs on the system ollama (11434); the newer 27B instance uses 11436.
# Override with LONGMEM_SERVER_PORT=11436 for the 27B run.
PORT="${LONGMEM_PORT:-11435}"
SERVER_PORT="${LONGMEM_SERVER_PORT:-11438}"

# Import writes use llama.cpp-specific thinking controls; reader/judge keep their
# own OpenAI-compatible reader configuration below.
export AI_GLASSES_LLM_PROVIDER=llama_cpp
export AI_GLASSES_LLM_MODEL="$MODEL"
# Direct mode (LONGMEM_DIRECT_HOST set) skips the SSH tunnel and hits the GPU
# box IP directly. Needed because the local sandbox/proxy mangles large requests
# sent through the localhost tunnel (502/404). llama-server must be bound 0.0.0.0.
if [ -n "${LONGMEM_DIRECT_HOST:-}" ]; then
  BASE_URL="http://${LONGMEM_DIRECT_HOST}:${SERVER_PORT}/v1"
else
  BASE_URL="http://localhost:${PORT}/v1"
fi
export AI_GLASSES_LLM_BASE_URL="$BASE_URL"
export AI_GLASSES_LLM_API_KEY=ollama
# Keep Ollama on a single GPU on the shared box (14B-Q4 ~9GB fits one A100).
export OLLAMA_NUM_GPUS=1

# Open tunnel Mac:PORT -> SERVER:SERVER_PORT; tear down on exit.
# Skipped when LONGMEM_DIRECT_HOST is set (direct IP mode, no tunnel).
if [ -z "${LONGMEM_DIRECT_HOST:-}" ]; then
  # - StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null: avoid writing
  #   ~/.ssh/known_hosts (sandbox/permissions may block it and destabilize ssh).
  # - ServerAliveInterval/CountMax: keep the tunnel alive across the long eval run.
  ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L "${PORT}:localhost:${SERVER_PORT}" "$SERVER" &
  TUNNEL_PID=$!
  cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; }
  trap cleanup EXIT
fi

if [ -n "${LONGMEM_DIRECT_HOST:-}" ]; then
  CHECK_URL="http://${LONGMEM_DIRECT_HOST}:${SERVER_PORT}/v1/models"
else
  CHECK_URL="http://localhost:${PORT}/v1/models"
fi
echo "Waiting for LLM server at $CHECK_URL ..."
for i in $(seq 1 20); do
  # /v1/models is OpenAI-compatible and works for both Ollama and llama.cpp.
  curl -fsS "$CHECK_URL" >/dev/null 2>&1 && break
  sleep 1
done

echo "==> Reader step (history_mode=$HISTORY_MODE, limit=$LIMIT, qtype=$QTYPE, model=$MODEL)"
READER_ARGS=(
  --reader-provider llama_cpp --reader-model "$MODEL"
  --reader-base-url "$BASE_URL" --reader-api-key ollama
  --history-mode "$HISTORY_MODE" --limit "$LIMIT"
  --output-dir "$OUT"
)
if [ -n "$QTYPE" ]; then
  READER_ARGS+=(--question-type "$QTYPE")
fi
# --no-cache: bypass the L2 per-question app_home/recall cache. The cache key now
# includes the app model + source hash (git HEAD + uncommitted diff), so it is safe
# to keep the cache on across model/code changes; this flag is for forced-fresh runs.
if [ -n "${LONGMEM_NO_CACHE:-}" ]; then
  READER_ARGS+=(--no-cache)
fi
# --workers: parallel question processing (safe — per-question import libraries are isolated).
READER_ARGS+=(--workers "$WORKERS")
# NOTE: no --overwrite. Always use a fresh output dir to avoid the runner's
# safety prompt that blocks bulk deletion of >50 existing case files.
# ``conda run`` captures child output by default, which hides the runner's
# stderr progress bar until the whole benchmark exits. Pass it through live.
PYTHONUNBUFFERED=1 conda run --no-capture-output -n hermes python -u \
  -m ai_glasses_memory_assistant.evals.longmemeval_runner "${READER_ARGS[@]}"

echo "==> Judge step"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n hermes python -u \
  scripts/judge_longmemeval.py --report-dir "$OUT" "$ORACLE"

echo "==> Done. Report: $OUT/eval-latest.md"
