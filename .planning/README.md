# Planning Files

本目录统一管理 `planning-with-files` skill 生成的工作文件。

## 目录规则

- 每个任务使用一个子目录：`.planning/<date>-<slug>/`。
- 每个子目录内保留该任务的 `task_plan.md`、`findings.md`、`progress.md`。
- `.planning/.active_plan` 记录当前活跃任务目录名。

## 新建计划

优先使用 skill 自带脚本创建多计划目录：

```bash
sh /Users/huyaokai/.codex/skills/planning-with-files/scripts/init-session.sh "任务名称"
```

查看或切换活跃计划：

```bash
sh /Users/huyaokai/.codex/skills/planning-with-files/scripts/set-active-plan.sh
sh /Users/huyaokai/.codex/skills/planning-with-files/scripts/set-active-plan.sh <plan_id>
```

不要把 `task_plan.md`、`findings.md`、`progress.md` 散放在仓库根目录。
