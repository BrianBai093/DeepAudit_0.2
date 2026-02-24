# Paper2Code (Development Stage) 

### 1) Project Overview

This project implements a 3-phase automated pipeline:

1. Phase 1: Extract reproducible paper facts from `Target/paper/full.md`, then generate `fingerprint`, `claims_ir`, and `task_spec`.  
2. Phase 2: Execute code inside E2B Sandbox and collect structured run outputs.  
3. Phase 3: Align evidence and verify claims, then generate final reports.

Current orchestration lives in `p2c/main.py` and `p2c/graph.py`.

### 2) Execution Strategy (Updated)

We have **dropped swe-agent** for execution and moved to the **official E2B + Codex** path:

- Runtime: E2B Sandbox
- Executor: `codex exec`
- Main agents: `prepare_sandbox` -> `run_codex_exec` -> `collect_codex_outputs`

### 3) What Works Today

- E2B sandbox creation works.
- Codex can run and write simple programs in sandbox.
- File and log collection works in simple scenarios.

### 4) Current Blockers

#### Issue A: Complex repo exits early with error code 1

For this long, multi-environment dependency repo, Codex often exits right after directory/file scanning with no meaningful error.  
From OpenAI API logs, execution often stops around:

```text
Function call
Arguments
shell({
  "command": [
    "/bin/sh",
    "-c",
    "cd /home/user/workspace/repo && head -n 20 requirements.txt"
  ]
})
```

Observed behavior: process exits after this step; `run_codex_exec` only gets `exit code 1`.

#### Issue B: Codex API token rate limit causes timeouts

- Current limit is around `20,000 tokens/min`.
- Complex repos frequently hit timeout/interruption.
- We likely need explicit wait/backoff throttling (manual or automated).

### 5) Current Investigation Focus

1. Improve Phase 2 observability: last command, stdout/stderr tail, pip tail, stage logs.  
2. Separate stage logs (main vs repair) to locate real failure point.  
3. Verify whether silent exits occur in:
   - Codex CLI layer
   - Sandbox command layer
   - API streaming layer (deadline/stream interruption)

### 6) TODO (Next Mandatory Task)

Build an isolated test module (independent from full pipeline complexity):

1. Create an E2B sandbox using the same method (Codex preinstalled).  
2. Inject key and upload target repo.  
3. Connect to the same sandbox via E2B CLI.  
4. Run commands manually and observe outputs/exit codes.  
5. Compare manual path vs agent path (command, cwd, env vars, timeout settings).

### 7) Goal of This Stage

Identify why the agent gets killed with no explicit error and provide a reproducible minimal case, then fix it:

- Complex repo should complete first-pass dependency and entrypoint execution reliably.  
- Failures must be diagnosable with explicit cause, not just `exit code 1`.

## 中文

### 1) 项目概述

本项目目标是实现一个三阶段自动化链路：

1. Phase 1：从论文（`Target/paper/full.md`）抽取可复现信息，生成 `fingerprint`、`claims_ir`、`task_spec`。  
2. Phase 2：在 E2B Sandbox 内执行代码并产出结构化运行结果。  
3. Phase 3：根据运行结果做证据对齐与结论验证，输出最终报告。

当前执行编排在 `p2c/main.py` 与 `p2c/graph.py`。

### 2) 当前技术路线（已变更）

已放弃旧的 swe-agent 执行路径，改为 **E2B 官方支持的 Codex** 路线：

- Runtime：E2B Sandbox
- 执行器：`codex exec`
- 主要 Agent：`prepare_sandbox` -> `run_codex_exec` -> `collect_codex_outputs`

### 3) 当前已验证能力

- E2B 可以正常创建 Sandbox。
- Codex 可以在 Sandbox 内执行并生成/运行简单程序。
- 简单场景下可以回传文件和日志。

### 4) 当前阻塞问题

#### 问题 A：复杂仓库执行直接退出（error code 1）

在当前多环境、长依赖链仓库中，Codex 经常在“目录/文件扫描阶段”后直接退出，未返回明确报错。  
根据 OpenAI API log，执行停在类似步骤：

```text
Function call
Arguments
shell({
  "command": [
    "/bin/sh",
    "-c",
    "cd /home/user/workspace/repo && head -n 20 requirements.txt"
  ]
})
```

表现：执行到此后无有效错误信息，进程退出，`run_codex_exec` 只拿到 `exit code 1`。

#### 问题 B：Codex API 速率限制导致超时

- 当前限制：约 `20000 tokens/min`  
- 在复杂仓库中容易触发超时或中断。  
- 需要加入更严格的节流/等待策略（人工或程序化 backoff）。

### 5) 当前排查方向

1. 强化 Phase 2 可观测性：记录最后命令、stdout/stderr tail、pip 日志、阶段日志。  
2. 细分主执行与修复阶段日志，区分 main/repair 的真实失败位置。  
3. 重点确认“无报错退出”是在：
   - Codex CLI 层
   - Sandbox 命令层
   - API streaming 层（例如 deadline/stream 中断）

### 6) TODO（下一步必须做）

实现一个独立测试模块（不依赖主 pipeline 复杂逻辑），流程如下：

1. 用同样方式创建 E2B Sandbox（预装 Codex）。  
2. 注入 Key 与测试 repo。  
3. 使用 E2B CLI 连接同一 Sandbox。  
4. 手动逐条执行命令并观察输出/退出码。  
5. 与 agent 自动执行路径对比差异（命令、cwd、环境变量、超时设置）。

### 7) 当前阶段目标

找出“agent 被杀死且无报错退出”的根因，并形成可复现最小案例，最终修复：

- 能稳定执行复杂仓库首轮依赖分析与入口命令。  
- 失败时必须有明确错误归因，而不是只有 `exit code 1`。  

---

