#!/usr/bin/env bash
# Stop local replay/test processes that currently call the llama server.
# Match established TCP connections, not a broad process-name pattern.
set -euo pipefail

LLAMA_PORT="${LONGMEM_LLAMA_PORT:-11438}"

find_pids() {
  lsof -t -nP -iTCP:"${LLAMA_PORT}" -sTCP:ESTABLISHED 2>/dev/null | sort -u || true
}

pids="$(find_pids)"
if [[ -z "${pids}" ]]; then
  echo "没有发现连接 llama 端口 ${LLAMA_PORT} 的测试进程。"
  exit 0
fi

echo "停止 llama 端口 ${LLAMA_PORT} 的测试进程: ${pids}"
kill -TERM ${pids} 2>/dev/null || true

for _ in {1..10}; do
  remaining="$(find_pids)"
  if [[ -z "${remaining}" ]]; then
    echo "已停止，llama 连接已清零。"
    exit 0
  fi
  sleep 1
done

remaining_pids="$(find_pids)"
echo "TERM 后仍有连接，发送 KILL: ${remaining_pids}"
kill -KILL ${remaining_pids} 2>/dev/null || true
sleep 1

if [[ -n "$(find_pids)" ]]; then
  echo "ERROR: 仍有进程连接 llama 端口 ${LLAMA_PORT}。" >&2
  exit 1
fi
echo "已停止，llama 连接已清零。"
