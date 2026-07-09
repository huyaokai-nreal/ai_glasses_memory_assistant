# 私人协作开发说明

本仓库是私人多人协作项目，不按公共开源项目维护 issue、社区规范或外部贡献流程。所有改动默认通过分支提交和代码 review 进入主分支。

## 开发流程

1. 从当前主分支创建功能分支，分支名尽量能看出目的，例如 `codex/refactor-audio-processing` 或 `dev/startup-docs`。
2. 修改前先读 `README.md`、`AGENTS.md`、`PLANS.md` 和 `docs/context/code-map.md`，确认入口、边界和验证命令。
3. 每次分支只解决一个清晰目标，不把无关重构、格式化和功能修改混在一起。
4. 涉及系统边界、启动方式、测试门禁或代码地图时，同步更新 `PLANS.md` 或 `docs/context/` 三文档。
5. 提交前确认没有真实密钥、本机证书、数据库、audit、缓存或 benchmark 数据进入 Git。

## 本地配置

推荐每个开发者使用自己的 home：

```bash
export AI_GLASSES_HOME="$HOME/.ai-glasses-memory-assistant"
```

复制示例配置后按本机情况填写：

```bash
mkdir -p "$AI_GLASSES_HOME"
cp .env.example "$AI_GLASSES_HOME/.env"
```

不要提交 `.env`、`.env.*`、`certs/`、本地 SQLite 数据库、`reports/` 下临时验证产物或包含真实用户数据的文件。

## 提交前验证

Python 代码改动至少运行：

```bash
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

文档或配置改动至少运行：

```bash
git diff --check
```

如果无法运行某条命令，在提交说明或 review 说明里写清楚原因和已做的替代检查。

## 代码边界

- `ai_glasses_memory_assistant/server.py` 是唯一 HTTP 入口，根目录 `server.py` 只是兼容薄入口。
- 聊天主链路在 `GlassesChatService.chat()`，不要绕过记忆写入门控、用户隔离、敏感信息保护或 timeline evidence。
- `tests/` 是核心保险丝，不要把旧的大型历史回归测试重新堆回默认门禁。
- 音频、speaker、ambient、新实验功能由负责同事补 focused 专项测试，不默认进入核心门禁。

## Commit message

推荐使用简洁具体的格式：

```text
refactor(audio): extract audio processing module
docs(dev): add private collaboration guide
test(startup): cover env fallback
fix(memory): avoid duplicate correction save
```
