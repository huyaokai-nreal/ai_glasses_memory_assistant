#!/usr/bin/env bash
# start_llama_watchdog.sh
#
# 在 10.252.17.5 (WX-AI-A100-01) 上以 tmux 守护 llama-server：
#   - 崩溃 / 退出后自动重启（直接解决"时间长了 llama-server 就断了"）
#   - tmux 会话在 ssh 断开后依然存活，不依赖本地终端
#
# 用法（在 Mac 上执行）:
#   bash scripts/start_llama_watchdog.sh
#
# 其内部逻辑（在服务器上运行）:
#   while true; do
#     if llama-server 在跑: sleep 30   # 健康, 继续保活
#     else: bash ~/run_llama.sh; sleep 5   # 没在跑, 拉起
#   done
#
# 该逻辑不依赖 run_llama.sh 是前台还是 nohup 后台：
#   - 若 run_llama.sh 前台跑 llama-server，则 while 循环阻塞到它崩溃再重启；
#   - 若 run_llama.sh 后台 nohup 拉起后返回，则下一轮 pgrep 会发现进程并进入 sleep。
#
# 查看状态:  ssh ykhu@10.252.17.5 'tmux ls; pgrep -af llama-server'
# 停止守护:  ssh ykhu@10.252.17.5 'tmux kill-session -t llama'
set -u

HOST="${LONGMEM_WATCHDOG_HOST:-ykhu@10.252.17.5}"
SESSION="${LONGMEM_WATCHDOG_SESSION:-llama}"

ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOST" bash -s <<REMOTE
  # 确保 tmux 可用
  command -v tmux >/dev/null 2>&1 || { echo "ERROR: 服务器无 tmux"; exit 1; }

  # 清理旧会话，重建干净的守护循环
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  sleep 1

  tmux new-session -d -s "$SESSION" "bash -c '
    while true; do
      if pgrep -x llama-server >/dev/null 2>&1; then
        sleep 30
      else
        echo \"[$(date +%F_%T)] llama-server 未运行, 启动中 ...\"
        bash ~/run_llama.sh
        sleep 5
      fi
    done
  '"

  sleep 3
  echo "tmux 会话 '$SESSION' 已创建 (llama-server 守护中)"
  tmux ls 2>/dev/null
  echo "---- 当前 llama-server 状态 ----"
  pgrep -ax llama-server || echo "(尚未就绪, 守护循环会在数秒内拉起)"
REMOTE
