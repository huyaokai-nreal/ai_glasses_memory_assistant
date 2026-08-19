pkill -9 -f "longmemeval_runner.py"
pkill -9 -f "run_longmemeval_server_llm.sh"
# 确认已停（无输出即已停止）
pgrep -af "longmemeval_runner.py" | grep -v grep || echo "已停止"
