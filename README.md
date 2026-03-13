# Paper2Code / DeepAudit

English and Chinese README for the current repository state.

---

## 1. What This Repository Is

### English

This repository is a three-phase pipeline for turning a paper plus its code repository into an auditable execution and verification trace.

High-level goal:

1. Read the paper and extract verifiable claims.
2. Inspect the repository and compile executable tasks.
3. Run the repository inside a sandboxed environment.
4. Compare observed evidence with the paper claims and produce a report.

The current implementation contains:

- A working Phase 1 pipeline for paper ingestion, fingerprint extraction, claim IR construction, repository analysis, and task compilation.
- Two Phase 2 implementations:
  - `legacy` phase 2: the older, heavier multi-stage Codex execution flow.
  - `new` phase 2: a newer E2B-first, single-`codex exec` flow added as an experimental parallel path.
- A Phase 3 verification/reporting pipeline that still expects legacy Phase 2 outputs.

### 中文

这个仓库实现的是一个三阶段流水线，用于把“论文 + 代码仓库”转换成一套可审计的执行与验证轨迹。

整体目标是：

1. 读取论文并抽取可验证 claim。
2. 分析代码仓库并生成可执行任务。
3. 在沙盒环境中运行仓库。
4. 将执行证据与论文 claim 对齐，生成审计报告。

当前实现状态包括：

- 一个可运行的 Phase 1，用于论文摄取、指纹抽取、claim IR 构建、仓库分析和任务编译。
- 两套 Phase 2：
  - `legacy` phase 2：旧的、较重的多阶段 Codex 执行流。
  - `new` phase 2：新增的、以 E2B 为中心的单次 `codex exec` 试验路径。
- 一个 Phase 3 验证/报告流水线，但它目前仍然依赖 legacy Phase 2 的产物。

---

## 2. Repository Layout

### English

Main directories:

- `p2c/`: core pipeline code
- `p2c/agents/phase1/`: paper ingestion and task compilation
- `p2c/agents/phase2/`: sandbox preparation and Codex execution
- `p2c/agents/phase3/`: evidence parsing, claim verification, report generation
- `p2c/runtime/`: runtime backends (`e2b` and `local`)
- `scripts/`: helper scripts, including E2B template builder
- `tests/`: targeted regression tests
- `Target/`: target repository under audit
- `artifacts/`: run outputs grouped by `run_id`
- `output/`: intermediate paper markdown output

Key files:

- [`p2c/main.py`](/mnt/e/DeepAudit_0.1/p2c/main.py): CLI entrypoint
- [`p2c/graph.py`](/mnt/e/DeepAudit_0.1/p2c/graph.py): phase orchestration and phase2 style routing
- [`p2c/io_artifacts.py`](/mnt/e/DeepAudit_0.1/p2c/io_artifacts.py): artifact tree and placeholders
- [`p2c/schemas.py`](/mnt/e/DeepAudit_0.1/p2c/schemas.py): Pydantic schemas

### 中文

主要目录：

- `p2c/`：核心流水线代码
- `p2c/agents/phase1/`：论文摄取与任务编译
- `p2c/agents/phase2/`：沙盒准备与 Codex 执行
- `p2c/agents/phase3/`：证据解析、claim 验证、报告生成
- `p2c/runtime/`：运行时后端（`e2b` 与 `local`）
- `scripts/`：辅助脚本，包括 E2B 模板构建脚本
- `tests/`：定向回归测试
- `Target/`：被审计的目标仓库
- `artifacts/`：按 `run_id` 组织的运行产物
- `output/`：论文 markdown 中间结果

关键文件：

- [main.py](/mnt/e/DeepAudit_0.1/p2c/main.py)：CLI 入口
- [graph.py](/mnt/e/DeepAudit_0.1/p2c/graph.py)：阶段编排与 phase2 风格路由
- [io_artifacts.py](/mnt/e/DeepAudit_0.1/p2c/io_artifacts.py)：产物树和占位文件
- [schemas.py](/mnt/e/DeepAudit_0.1/p2c/schemas.py)：Pydantic 数据结构

---

## 3. Pipeline Overview

### Phase 1

#### English

Phase 1 produces the execution plan from the paper and repository.

It runs these agents in order:

1. `ingest_paper`
2. `extract_fingerprint_guide`
3. `extract_fingerprint_atomic`
4. `extract_fingerprint_filter`
5. `build_claims_ir`
6. `repo_analysis`
7. `compile_task_spec`

Important outputs:

- `fingerprint/fingerprint.json`
- `fingerprint/claims_ir.json`
- `task/repo_analysis.json`
- `task/task_spec.json`
- `task/metric_contract.json`

#### 中文

Phase 1 的作用是从论文和代码仓库生成执行计划。

它按顺序运行以下 agent：

1. `ingest_paper`
2. `extract_fingerprint_guide`
3. `extract_fingerprint_atomic`
4. `extract_fingerprint_filter`
5. `build_claims_ir`
6. `repo_analysis`
7. `compile_task_spec`

重要输出包括：

- `fingerprint/fingerprint.json`
- `fingerprint/claims_ir.json`
- `task/repo_analysis.json`
- `task/task_spec.json`
- `task/metric_contract.json`

### Phase 2

#### English

Phase 2 executes the repository in a runtime backend.

Two implementations exist today:

- `legacy` style
  - default path
  - multi-stage
  - heavier local orchestration
  - produces legacy outputs such as `run_manifest.json` and `claim_alignment.json`
- `new` style
  - selected via `P2C_PHASE2_STYLE=new`
  - single `codex exec`
  - much thinner local orchestration
  - primary fact artifact is `execution_summary.json`

#### 中文

Phase 2 负责在运行时后端中执行目标仓库。

目前存在两套实现：

- `legacy` 风格
  - 默认路径
  - 多阶段执行
  - 本地编排逻辑较重
  - 会产出 `run_manifest.json`、`claim_alignment.json` 等旧风格文件
- `new` 风格
  - 通过 `P2C_PHASE2_STYLE=new` 启用
  - 单次 `codex exec`
  - 本地编排更薄
  - 核心事实文件是 `execution_summary.json`

### Phase 3

#### English

Phase 3 observes metrics, aligns evidence, verifies claims, and writes a report.

Current Phase 3 still assumes legacy Phase 2 outputs.

#### 中文

Phase 3 负责观察指标、对齐证据、验证 claim，并生成报告。

当前 Phase 3 仍然假设上游使用的是 legacy Phase 2 输出。

---

## 4. Phase 2 Styles: Legacy vs New

### English

`legacy` Phase 2:

- Files:
  - [`prepare_sandbox.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/prepare_sandbox.py)
  - [`run_codex_exec.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/run_codex_exec.py)
  - [`collect_codex_outputs.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/collect_codex_outputs.py)
- Behavior:
  - multi-stage discovery / execution / repair
  - expects many intermediate artifacts
  - still tied to legacy manifest-style outputs

`new` Phase 2:

- Files:
  - [`prepare_sandbox_newstyle.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/prepare_sandbox_newstyle.py)
  - [`run_codex_exec_newstyle.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/run_codex_exec_newstyle.py)
  - [`collect_codex_outputs_newstyle.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/collect_codex_outputs_newstyle.py)
  - [`codex_prompt_templates_newstyle.py`](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/codex_prompt_templates_newstyle.py)
- Behavior:
  - one Codex session
  - task spec is the main explicit input
  - summary-driven result collection
  - designed to reduce local over-control

Routing is controlled in [`graph.py`](/mnt/e/DeepAudit_0.1/p2c/graph.py).

### 中文

`legacy` Phase 2：

- 文件：
  - [prepare_sandbox.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/prepare_sandbox.py)
  - [run_codex_exec.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/run_codex_exec.py)
  - [collect_codex_outputs.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/collect_codex_outputs.py)
- 行为：
  - discovery / execution / repair 多阶段
  - 需要很多中间产物
  - 仍然绑定 legacy 的 manifest 风格输出

`new` Phase 2：

- 文件：
  - [prepare_sandbox_newstyle.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/prepare_sandbox_newstyle.py)
  - [run_codex_exec_newstyle.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/run_codex_exec_newstyle.py)
  - [collect_codex_outputs_newstyle.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/collect_codex_outputs_newstyle.py)
  - [codex_prompt_templates_newstyle.py](/mnt/e/DeepAudit_0.1/p2c/agents/phase2/codex_prompt_templates_newstyle.py)
- 行为：
  - 单次 Codex 会话
  - `task_spec` 是主要显式输入
  - 以 summary 为中心做结果收集
  - 目标是减少本地过度编排

路由逻辑在 [graph.py](/mnt/e/DeepAudit_0.1/p2c/graph.py) 中。

---

## 5. Runtime Backends

### English

Supported backends:

- `e2b` (default)
- `local`

Runtime factory:

- [`factory.py`](/mnt/e/DeepAudit_0.1/p2c/runtime/factory.py)

E2B runtime:

- [`e2b_runtime.py`](/mnt/e/DeepAudit_0.1/p2c/runtime/e2b_runtime.py)

Current default E2B template:

- `openai-codex`

Optional explicit template override:

- `P2C_E2B_TEMPLATE=<template_name>`

### 中文

支持的运行时后端：

- `e2b`（默认）
- `local`

运行时工厂：

- [factory.py](/mnt/e/DeepAudit_0.1/p2c/runtime/factory.py)

E2B 运行时实现：

- [e2b_runtime.py](/mnt/e/DeepAudit_0.1/p2c/runtime/e2b_runtime.py)

当前默认 E2B 模板：

- `openai-codex`

如需显式覆盖模板：

- `P2C_E2B_TEMPLATE=<template_name>`

---

## 6. Environment Variables

### English

Common variables:

- `OPENAI_API_KEY`: required for LLM calls and Codex inside sandbox
- `OPENAI_MODEL`: model for the local paper-processing LLM client, default `gpt-5.1`
- `OPENAI_BASE_URL`: optional OpenAI-compatible base URL
- `E2B_API_KEY`: required when using `P2C_RUNTIME_BACKEND=e2b`
- `P2C_RUNTIME_BACKEND`: `e2b` or `local`, default `e2b`
- `P2C_PHASE2_STYLE`: `legacy` or `new`, default `legacy`
- `P2C_CODEX_MODEL`: Codex CLI model, default `gpt-5.1`
- `P2C_E2B_TEMPLATE`: optional explicit E2B template name

### 中文

常用环境变量：

- `OPENAI_API_KEY`：本地 LLM 调用以及沙盒内 Codex 都需要
- `OPENAI_MODEL`：本地论文处理 LLM 模型，默认 `gpt-5.1`
- `OPENAI_BASE_URL`：可选的 OpenAI 兼容接口地址
- `E2B_API_KEY`：使用 `P2C_RUNTIME_BACKEND=e2b` 时需要
- `P2C_RUNTIME_BACKEND`：`e2b` 或 `local`，默认 `e2b`
- `P2C_PHASE2_STYLE`：`legacy` 或 `new`，默认 `legacy`
- `P2C_CODEX_MODEL`：Codex CLI 使用的模型，默认 `gpt-5.1`
- `P2C_E2B_TEMPLATE`：可选的 E2B 模板名覆盖

---

## 7. Installation

### English

This repository currently does not include a pinned top-level environment file for the whole orchestrator itself. At minimum you need:

- Python 3.10+
- `pydantic`
- `pytest` for tests
- E2B SDK for sandbox runs

Typical minimal setup:

```bash
python3 -m pip install -U pydantic pytest e2b tomli
```

### 中文

当前仓库还没有为“整个 orchestrator 自身”提供一份固定顶层环境文件。最少需要：

- Python 3.10+
- `pydantic`
- 运行测试用的 `pytest`
- 沙盒运行所需的 E2B SDK

最小安装示例：

```bash
python3 -m pip install -U pydantic pytest e2b tomli
```

---

## 8. How To Run

### Phase 1

#### English

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md "/abs/path/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

#### 中文

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md "/abs/path/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

### Phase 2 Legacy

#### English

```bash
python -m p2c.main \
  --phase 2 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

#### 中文

```bash
python -m p2c.main \
  --phase 2 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

### Phase 2 New Style

#### English

```bash
P2C_PHASE2_STYLE=new python -m p2c.main \
  --phase 2 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

#### 中文

```bash
P2C_PHASE2_STYLE=new python -m p2c.main \
  --phase 2 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

### Phase 3

#### English

Run Phase 3 only after a legacy-style Phase 2 run has produced the required files.

```bash
python -m p2c.main \
  --phase 3 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

#### 中文

只有在 legacy 风格的 Phase 2 已经产出所需文件后，才能运行 Phase 3。

```bash
python -m p2c.main \
  --phase 3 \
  --paper_md "/abs/path/output/paper.md" \
  --paper_md_out "/abs/path/output/paper.md" \
  --repo_dir "/abs/path/Target" \
  --run_id demo_run \
  --artifacts_dir "/abs/path/artifacts" \
  --budget_minutes 60
```

---

## 9. Artifact Layout

### English

Per-run outputs live under:

- `artifacts/<run_id>/`

Important subpaths:

- `fingerprint/`: extracted claim and paper artifacts
- `task/`: repository analysis and execution task spec
- `execution/`: runtime logs and Codex outputs
- `results/`: metrics, verdicts, and final report

Examples:

- `artifacts/<run_id>/task/task_spec.json`
- `artifacts/<run_id>/execution/codex_outputs/execution_summary.json`
- `artifacts/<run_id>/results/verdict.json`
- `artifacts/<run_id>/results/report.md`

### 中文

每次运行的输出位于：

- `artifacts/<run_id>/`

重要子目录：

- `fingerprint/`：论文与 claim 抽取产物
- `task/`：仓库分析与执行任务定义
- `execution/`：运行时日志和 Codex 输出
- `results/`：指标、结论与最终报告

例如：

- `artifacts/<run_id>/task/task_spec.json`
- `artifacts/<run_id>/execution/codex_outputs/execution_summary.json`
- `artifacts/<run_id>/results/verdict.json`
- `artifacts/<run_id>/results/report.md`

---

## 10. Testing

### English

Targeted tests currently available:

- [`tests/test_phase2_e2b_first.py`](/mnt/e/DeepAudit_0.1/tests/test_phase2_e2b_first.py)
- [`tests/test_phase2_newstyle.py`](/mnt/e/DeepAudit_0.1/tests/test_phase2_newstyle.py)

Run the new-style tests:

```bash
python3 -m pytest tests/test_phase2_newstyle.py -q -s
```

Run the narrower phase2 regression tests:

```bash
python3 -m pytest tests/test_phase2_e2b_first.py -q -s
```

### 中文

当前可用的定向测试主要有：

- [tests/test_phase2_e2b_first.py](/mnt/e/DeepAudit_0.1/tests/test_phase2_e2b_first.py)
- [tests/test_phase2_newstyle.py](/mnt/e/DeepAudit_0.1/tests/test_phase2_newstyle.py)

运行 newstyle 测试：

```bash
python3 -m pytest tests/test_phase2_newstyle.py -q -s
```

运行 phase2 的定向回归测试：

```bash
python3 -m pytest tests/test_phase2_e2b_first.py -q -s
```

---

## 11. Current Known Issues / Unresolved Problems

### English

This section describes the current known problems as of the repository state today.

1. Phase 3 is still legacy-coupled.
   - [`main.py`](/mnt/e/DeepAudit_0.1/p2c/main.py) and Phase 3 prerequisites still require:
     - `execution/codex_outputs/run_manifest.json`
     - `execution/codex_outputs/claim_alignment.json`
   - As a result, the new-style Phase 2 is not yet a drop-in replacement for Phase 3.

2. Legacy Phase 2 and new Phase 2 coexist, but the artifact contract is not unified.
   - Legacy flow expects many intermediate files.
   - New flow treats `execution_summary.json` as the primary source of truth.
   - The repository currently supports both, but not with a single clean contract.

3. `ArtifactManager` still creates many legacy placeholders.
   - [`io_artifacts.py`](/mnt/e/DeepAudit_0.1/p2c/io_artifacts.py) initializes old files even when the new-style phase2 path is used.
   - This is compatible, but it makes the artifact tree noisy and conceptually inconsistent.

4. The top-level environment setup is still incomplete.
   - There is no canonical root `requirements.txt` or `pyproject.toml` for the orchestrator itself.
   - Installation is currently based on inferred minimal dependencies.

5. E2B runtime support is practical but not yet fully normalized.
   - Default template is `openai-codex`.
   - Extra tools such as `Rscript` may require runtime bootstrap or a custom template.
   - This means some repositories still hit environment-dependent behavior.

6. Legacy Phase 2 remains prone to over-control.
   - It still uses a heavier local orchestration style and multiple artifact expectations.
   - The new-style path was added specifically to experiment with a thinner contract.

7. Sandbox code modifications by Codex are still possible.
   - The current prompt allows repository execution and debugging.
   - Depending on the task and failure mode, Codex may modify files in the sandbox while trying to make the repo runnable.
   - This behavior is visible in execution logs and may not be desirable for every audit mode.

8. Documentation and workflow are still evolving.
   - This README describes the current repository state, not a frozen stable release.

### 中文

本节描述的是当前仓库状态下尚未解决的问题。

1. Phase 3 仍然绑定 legacy 输出。
   - [main.py](/mnt/e/DeepAudit_0.1/p2c/main.py) 以及 Phase 3 前置检查仍然要求：
     - `execution/codex_outputs/run_manifest.json`
     - `execution/codex_outputs/claim_alignment.json`
   - 因此，newstyle Phase 2 目前还不能无缝替代 Phase 3 的上游。

2. legacy Phase 2 与 new Phase 2 并存，但产物契约还没有统一。
   - legacy 流程依赖很多中间文件。
   - newstyle 流程把 `execution_summary.json` 当作主要真相源。
   - 当前仓库同时支持两者，但还不是一个统一、干净的接口。

3. `ArtifactManager` 仍然会创建大量 legacy 占位文件。
   - [io_artifacts.py](/mnt/e/DeepAudit_0.1/p2c/io_artifacts.py) 即使在使用 newstyle phase2 时，也会初始化旧文件。
   - 这在兼容性上没有问题，但会让产物树显得嘈杂，概念上也不完全一致。

4. 顶层环境配置仍不完整。
   - orchestrator 自身还没有一份规范的根级 `requirements.txt` 或 `pyproject.toml`。
   - 当前安装方式仍然依赖“最小依赖推断”。

5. E2B 运行时支持可用，但还没有完全标准化。
   - 默认模板是 `openai-codex`。
   - 像 `Rscript` 这样的额外工具，可能仍然需要运行时 bootstrap 或自定义模板。
   - 因此某些仓库仍然会受到环境差异影响。

6. legacy Phase 2 仍然容易“本地编排过重”。
   - 它仍然保留了较重的本地控制逻辑和多种旧产物要求。
   - newstyle 路径正是为了试验更薄的契约而新增的。

7. Codex 仍可能在 sandbox 中修改代码。
   - 当前提示词允许它为执行和调试做必要尝试。
   - 在某些任务和失败场景下，Codex 可能会在 sandbox 中修改仓库文件，以尝试让项目运行起来。
   - 这种行为可以在执行日志中看到，但对某些审计模式来说并不理想。

8. 文档与工作流仍在演进中。
   - 本 README 描述的是“当前仓库状态”，不是一个完全冻结的稳定版本。

---

## 12. Recommended Next Steps

### English

Recommended engineering cleanup:

1. Decide whether new-style Phase 2 should become the default.
2. If yes, refactor Phase 3 to consume `execution_summary.json` instead of legacy-only artifacts.
3. Add a root environment file for the orchestrator.
4. Reduce `ArtifactManager` placeholder noise for new-style runs.
5. Add end-to-end tests covering:
   - Phase 1 -> Phase 2 new
   - Phase 2 new -> Phase 3 compatibility or explicit incompatibility handling

### 中文

推荐的后续工程化动作：

1. 明确决定是否让 newstyle Phase 2 成为默认路径。
2. 如果是，就把 Phase 3 改成消费 `execution_summary.json`，而不是只认 legacy 产物。
3. 为 orchestrator 本身补一份根级环境文件。
4. 减少 newstyle 运行时 `ArtifactManager` 的 legacy 占位噪声。
5. 增加端到端测试，覆盖：
   - Phase 1 -> Phase 2 new
   - Phase 2 new -> Phase 3 兼容或显式不兼容处理

---

## 13. Status Summary

### English

Current status in one sentence:

This repository is functional for Phase 1, has both legacy and experimental new-style Phase 2 implementations, and still requires additional unification work before the full pipeline is cleanly consistent end to end.

### 中文

用一句话概括当前状态：

这个仓库目前的 Phase 1 是可运行的，Phase 2 同时存在 legacy 和实验性的 newstyle 两套实现，但在全链路一致性方面仍然需要进一步收敛和统一。
