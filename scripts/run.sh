
# 快速冒烟：1 题验证通路。缓存 key 已含 model + 代码 hash，默认走缓存；
# 若想强制重测 import/recall（测性能），再加 LONGMEM_NO_CACHE=1。
LONGMEM_DIRECT_HOST=10.252.17.5 LONGMEM_WORKERS=1 \
bash scripts/run_longmemeval_server_llm.sh 1 import
