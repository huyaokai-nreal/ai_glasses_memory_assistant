#!/usr/bin/env bash
# resume_eval_500.sh
#
# 续跑 342/500 题 (pref500-20260812-0ef905):
#   - runner 加 --resume: 扫描每个 case 的 completed.json, 已完成 342 题自动跳过,
#     只补跑剩余 ~158 题, 然后跑官方 judge 补正式命中率。
#   - 前台运行, 由调用方以后台任务方式拉起(进程保活); 日志落盘到 <OUT>-resume.log。
#
# 用法（在 Mac 上执行, 需先确认 10.252.17.5:11438 上的 llama-server 已起）:
#   bash scripts/resume_eval_500.sh
#
# 进度查看:
#   （scripts/longmemeval_progress.sh 不存在，用下面这行即可）
#   tail -f reports/longmemeval/pref500-20260812-0ef905-resume.log
set -u

cd "$(dirname "$0")/.."

OUT="reports/longmemeval/pref500-20260812-0ef905"   # 修改此输出目录！！！！
HOST_IP="${LONGMEM_DIRECT_HOST:-10.252.17.5}"
PORT="${LONGMEM_SERVER_PORT:-11438}"
WORKERS="${LONGMEM_WORKERS:-4}"   # run_llama.sh 起 4 slots, 4 路并行对齐上限; 原 run.sh 用 1

# 整条链路(import 写记忆 + judge 打分)都走本地 qwen3.8-27B, 彻底脱离 DeepSeek。
# 必须显式设 AI_GLASSES_LLM_API_KEY=ollama, 否则 judge 会回退到 DEEPSEEK_API_KEY 兜底。
# 复用本脚本已有的 HOST_IP/PORT, Mac 可直连 10.252.17.5:11438, 无需 SSH 隧道。
export AI_GLASSES_LLM_PROVIDER=llama_cpp
export AI_GLASSES_LLM_MODEL=qwen3.8-27b-32k
export AI_GLASSES_LLM_BASE_URL="http://${HOST_IP}:${PORT}/v1"
export AI_GLASSES_LLM_API_KEY=ollama

# 前置检查: 服务端是否在线
if ! curl -fsS --max-time 5 "http://${HOST_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: llama-server @ ${HOST_IP}:${PORT} 不在线, 请先 bash scripts/start_llama_watchdog.sh"
  exit 1
fi

LOG="${OUT}-resume.log"
# 前台运行(由调用方以后台任务方式拉起, 保证进程不被回收); 全部输出落盘到 LOG。
exec > "$LOG" 2>&1
set -e
echo "[$(date +%F_%T)] resume reader (--resume, skip completed) ..."
conda run -n hermes python -m ai_glasses_memory_assistant.evals.longmemeval_runner \
  --reader-provider llama_cpp --reader-model qwen3.8-27b-32k \
  --reader-base-url "http://${HOST_IP}:${PORT}/v1" --reader-api-key ollama \
  --history-mode import --limit 0 --output-dir "$OUT" --no-cache --workers "$WORKERS" --resume

echo "[$(date +%F_%T)] judge step ..."
conda run -n hermes python scripts/judge_longmemeval.py --report-dir "$OUT" data/benchmarks/longmemeval/longmemeval_oracle.json

echo "[$(date +%F_%T)] Done. Report: $OUT/eval-latest.md"
