# 测试说明

`tests/` 是当前代码库的默认单元测试门禁，故意保持很小。

运行命令：

```bash
conda run -n hermes python -m pytest tests -q
```

## 核心保险丝

默认测试只保护最核心、最不能坏的边界：

- `test_core_startup.py`：启动、app home/env、server 入口、文档入口合同。
- `test_core_storage.py`：SQLite 用户隔离、timeline 脱敏、memory job 持久化。
- `test_core_chat.py`：`GlassesChatService.chat()` 回复、记忆写入、敏感信息拒存、按用户隔离召回。

## 不放回默认门禁的内容

不要把旧的大型回归测试重新堆回这个目录。尤其不要把下面这些内容放进默认单元测试门禁：

- 音频、speaker、emotion、wakeword、ambient runtime 实验。
- 过细的 planner 短语规则或历史行为补丁。
- 大型 eval harness 内部实现。
- 一次性开发记录或迁移历史。

这些方向如果有负责人正在改，可以补 focused 专项测试；但不要让默认门禁重新变成大型历史档案。

## 新增测试原则

新增测试前，先判断它是不是在保护核心安全边界。不是的话，优先放到 focused 功能测试、live eval 场景或负责人自己的验证流程里。

测试应短、可读，尽量从 service 层验证。不要再创建新的巨型 policy 测试文件。
