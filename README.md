# Paper2Code (Development Stage) / 当前开发阶段说明

## 中文

### 1) 项目概述

本项目目标是实现一个三阶段自动化链路：

1. Phase 1：从论文（`Target/paper/full.md`）抽取可复现信息，生成 `fingerprint`、`claims_ir`、`task_spec`。  
2. Phase 2：在 E2B Sandbox 内执行代码并产出结构化运行结果。  
3. Phase 3：根据运行结果做证据对齐与结论验证，输出最终报告。

当前执行编排在 `p2c/main.py` 与 `p2c/graph.py`。

### 1.1) 当前代码架构（已整理）

Agent 已按阶段拆分到三个子目录：

- `p2c/agents/phase1/`
  - `ingest_paper.py`
  - `extract_fingerprint_guide.py`
  - `extract_fingerprint_atomic.py`
  - `extract_fingerprint_filter.py`
  - `build_claims_ir.py`
  - `compile_task_spec.py`
- `p2c/agents/phase2/`
  - `prepare_sandbox.py`
  - `run_codex_exec.py`
  - `collect_codex_outputs.py`
  - `codex_exec_support.py`
  - `codex_prompt_templates.py`
- `p2c/agents/phase3/`
  - `observe_metrics.py`
  - `align_evidence.py`
  - `verify_claims.py`
  - `audit_report.py`

公共基类仍在 `p2c/agents/base.py`，编排入口仍在 `p2c/graph.py`。

### 2) 当前技术路线（已变更）

已放弃旧的 swe-agent 执行路径，改为 **E2B 官方支持的 Codex** 路线：

- Runtime：E2B Sandbox
- 执行器：`codex exec`
- 主要 Agent：`prepare_sandbox` -> `run_codex_exec` -> `collect_codex_outputs`

### 3) 当前已验证能力

- E2B 可以正常创建 Sandbox。
- Codex 可以在 Sandbox 内执行并生成/运行简单程序。
- 简单场景下可以回传文件和日志。

### 3.1) Phase 2 依赖能力链路（最新）

Phase 2 现在按下面顺序处理依赖能力：

1. 模板预装能力优先（推荐）  
   使用 `scripts/build_e2b_codex_template.py` 构建自定义模板，并通过 `P2C_E2B_TEMPLATE` 指定。
2. 运行时 sudo apt 兜底  
   若 runtime 缺 pip，先探测 `sudo -n true`，可用时走 `sudo apt-get update/install`。
3. legacy 兼容替代（默认开启）  
   对 `tensorflow==1.15.4`、`numpy==1.13.3` 等旧 pin 生成 `requirements.compat.txt` 并重试安装。
4. 若依赖仍不可用  
   不进入 Codex main，执行 entrypoint probe 并写完整 fallback outputs + `execution/codex_failure.json`。

关键环境变量：

- `P2C_E2B_TEMPLATE`（默认 `openai-codex`）
- `P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE`（默认 `1`）
- `P2C_DEP_COMPAT_MODE`（默认 `1`）
- `P2C_DEP_COMPAT_PROFILE`（默认 `tf1_legacy`）
- `P2C_DEP_BOOTSTRAP_ENABLE`（默认 `1`）
- `P2C_DEP_BOOTSTRAP_APT_ENABLE`（默认 `1`）

### 3.2) 当前运行状态（2026-02）

- Phase 2 目前已可在真实仓库上“跑起来并进入训练/执行阶段”。  
- 但整体耗时仍偏长（依赖准备 + 大仓库扫描 + 长任务执行）。  
- OpenAI API 在高负载时容易触发速率上限（尤其 TPM），导致中断、重试或总时长进一步拉长。  

短期建议：

1. 降低单次 prompt 与日志输出量（避免打印大 JSON 和超长目录列表）。  
2. 先用更小 run_matrix 做冒烟，再跑完整配置。  
3. 在 runner 侧继续增强限流退避与分批执行策略。  
4. 优先使用预装依赖的自定义模板，减少运行时安装开销。

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

### 5.1) 启动超时修复（已实现）

Phase 2 的 `run_codex_exec` 已增加启动器健壮性处理：

1. 背景启动命令使用 `timeout=0`（禁用 E2B 请求级超时）。  
2. 若仍出现 `context deadline exceeded`，不会立即判失败；会先探测 `pid/rc` 文件确认后台进程是否已启动。  
3. 只有在“超时且未探测到任何后台进程/退出码”时才判定 `CODEX_BACKGROUND_LAUNCH_FAILED`。

### 5.2) 仍待优化项（性能/配额）

1. 长任务的总耗时仍偏高，尤其在 legacy 依赖仓库上。  
2. API 速率限制（TPM）仍是主要不稳定因素之一。  
3. 后续会继续做 prompt 压缩、阶段拆分与更保守的重试节奏。

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

## English

### 1) Project Overview

This project implements a 3-phase automated pipeline:

1. Phase 1: Extract reproducible paper facts from `Target/paper/full.md`, then generate `fingerprint`, `claims_ir`, and `task_spec`.  
2. Phase 2: Execute code inside E2B Sandbox and collect structured run outputs.  
3. Phase 3: Align evidence and verify claims, then generate final reports.

Current orchestration lives in `p2c/main.py` and `p2c/graph.py`.

### 1.1) Current Code Layout (Refactored)

Agents are now organized by phase:

- `p2c/agents/phase1/`
  - `ingest_paper.py`
  - `extract_fingerprint_guide.py`
  - `extract_fingerprint_atomic.py`
  - `extract_fingerprint_filter.py`
  - `build_claims_ir.py`
  - `compile_task_spec.py`
- `p2c/agents/phase2/`
  - `prepare_sandbox.py`
  - `run_codex_exec.py`
  - `collect_codex_outputs.py`
  - `codex_exec_support.py`
  - `codex_prompt_templates.py`
- `p2c/agents/phase3/`
  - `observe_metrics.py`
  - `align_evidence.py`
  - `verify_claims.py`
  - `audit_report.py`

Shared base agent remains in `p2c/agents/base.py`, while orchestration remains in `p2c/graph.py`.

### 2) Execution Strategy (Updated)

We have **dropped swe-agent** for execution and moved to the **official E2B + Codex** path:

- Runtime: E2B Sandbox
- Executor: `codex exec`
- Main agents: `prepare_sandbox` -> `run_codex_exec` -> `collect_codex_outputs`

### 3) What Works Today

- E2B sandbox creation works.
- Codex can run and write simple programs in sandbox.
- File and log collection works in simple scenarios.

### 3.1) Phase 2 Dependency Capability Flow (Latest)

Phase 2 now resolves dependencies in this order:

1. Template preinstall first (recommended)  
   Build a custom template with `scripts/build_e2b_codex_template.py`, then set `P2C_E2B_TEMPLATE`.
2. Runtime sudo apt fallback  
   If pip is missing, probe `sudo -n true`; when available, run `sudo apt-get update/install`.
3. Legacy compatibility fallback (enabled by default)  
   For old pins such as `tensorflow==1.15.4` and `numpy==1.13.3`, generate `requirements.compat.txt` and retry.
4. If dependencies are still unresolved  
   Skip Codex main, run one probe per entrypoint, and emit fallback outputs plus `execution/codex_failure.json`.

Key environment variables:

- `P2C_E2B_TEMPLATE` (default `openai-codex`)
- `P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE` (default `1`)
- `P2C_DEP_COMPAT_MODE` (default `1`)
- `P2C_DEP_COMPAT_PROFILE` (default `tf1_legacy`)
- `P2C_DEP_BOOTSTRAP_ENABLE` (default `1`)
- `P2C_DEP_BOOTSTRAP_APT_ENABLE` (default `1`)

### 3.2) Current Runtime Status (2026-02)

- Phase 2 can now run end-to-end far enough to enter real training/execution paths on complex repos.  
- Runtime is still long in many cases (dependency bootstrap + repo scan + long-running workloads).  
- API rate limits (especially TPM) are still a practical bottleneck and can trigger retries/timeouts.

Short-term recommendations:

1. Keep prompts and logs compact (avoid dumping large JSON/files).  
2. Run a small smoke run_matrix before full runs.  
3. Continue improving runner-side backoff and staged execution.  
4. Prefer custom preinstalled templates to reduce runtime install overhead.

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

### 5.1) Launcher Timeout Fix (Implemented)

`run_codex_exec` now has launcher hardening:

1. Background launcher call uses `timeout=0` to disable request-level timeout in E2B command RPC.  
2. On `context deadline exceeded`, runner probes `pid/rc` before failing, to avoid false negatives when the process already started.  
3. The stage is marked failed only if timeout happened and no running/exited process can be confirmed.

### 5.2) Remaining Optimization Work (Performance/Quota)

1. End-to-end runtime is still high on legacy-heavy repositories.  
2. API TPM limits remain a key instability source.  
3. Next iterations will focus on prompt compression, stage splitting, and more conservative retry pacing.

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
