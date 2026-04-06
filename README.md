# DeepAudit v0.5

> 论文可复现性自动审计系统 — Automated Paper Reproducibility Audit System

---

## 1. 项目简介 / What Is This

### English

DeepAudit is a three-phase pipeline that takes a research paper and its companion code repository, then automatically:

1. **Phase 1** — Reads the paper, extracts verifiable claims, analyzes the repository, and compiles executable tasks.
2. **Phase 2** — Provisions a conda environment, executes the repository step-by-step via **Claude Code Agent SDK**, self-heals on failure, and collects metrics.
3. **Phase 3** — Aligns execution evidence with paper claims, verifies reproducibility within tolerance, and produces a verdict report.

**v0.5 highlights**: Phase 2 execution engine migrated from OpenAI Codex CLI to Claude Code Agent SDK. Claude Code is now the **primary and only executor** — every step goes through Claude Code's Bash tool with built-in self-healing and iteration capabilities. This follows the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) design philosophy where the AI agent itself is the executor.

### 中文

DeepAudit 是一个三阶段流水线，输入一篇论文及其配套代码仓库，自动完成：

1. **Phase 1** — 阅读论文、抽取可验证 claim、分析仓库结构、编译可执行任务。
2. **Phase 2** — 创建 conda 环境，通过 **Claude Code Agent SDK** 逐步执行仓库代码，失败时自动修复，收集指标。
3. **Phase 3** — 将执行证据与论文 claim 对齐，在容差范围内验证可复现性，生成审计报告。

**v0.5 重点变更**：Phase 2 执行引擎从 OpenAI Codex CLI 迁移至 Claude Code Agent SDK。Claude Code 现在是**唯一首选执行方式**——所有步骤统一通过 Claude Code 的 Bash tool 执行，自带 self-heal 和迭代能力。设计理念参照 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)，AI agent 本身就是执行者。

---

## 2. 目录结构 / Repository Layout

```
DeepAudit_0.2/
├── p2c/                            # 核心流水线 / Core pipeline
│   ├── main.py                     # CLI 入口 / CLI entry point
│   ├── graph.py                    # 阶段编排 / Phase orchestration
│   ├── schemas.py                  # Pydantic 数据模型 / Data models
│   ├── io_artifacts.py             # 产物管理 / Artifact tree manager
│   ├── failure_taxonomy.py         # 失败分类体系 / Failure classification (v2)
│   ├── agents/
│   │   ├── base.py                 # BaseAgent 抽象基类
│   │   ├── phase1/                 # 论文摄取与任务编译
│   │   │   ├── ingest_paper.py
│   │   │   ├── extract_fingerprint_guide.py
│   │   │   ├── extract_fingerprint_atomic.py
│   │   │   ├── extract_fingerprint_filter.py
│   │   │   ├── build_claims_ir.py
│   │   │   ├── repo_analysis.py
│   │   │   └── compile_task_spec.py
│   │   ├── phase2/                 # 环境配置与执行
│   │   │   ├── orchestrator.py     # 状态机编排器
│   │   │   ├── planner.py          # 执行计划生成
│   │   │   ├── tool_agent.py       # 环境配置 (conda/pip)
│   │   │   ├── codex_executor.py   # Claude Code 执行引擎 (v0.5)
│   │   │   ├── local_prompt_templates.py
│   │   │   └── result_extraction.py
│   │   └── phase3/                 # 验证与报告
│   │       ├── observe_metrics.py
│   │       ├── align_evidence.py
│   │       ├── verify_claims.py
│   │       └── audit_report.py
│   ├── llm/
│   │   └── client.py              # OpenAI 兼容 LLM 客户端
│   ├── runtime/
│   │   └── conda_env.py           # Conda 环境管理
│   └── utils/
│       └── console.py             # 日志与格式化
├── tests/                          # 回归测试
├── Target/                         # 被审计的目标仓库
├── artifacts/                      # 运行产物 (按 run_id 分组)
├── output/                         # 论文 markdown 中间结果
├── scripts/                        # 辅助脚本
└── requirements.txt                # Python 依赖
```

---

## 3. 架构概览 / Architecture Overview

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│                        Phase 1                          │
│  Paper + Repo → Claims IR + Task Spec + Metric Contract │
│                                                         │
│  ingest_paper → extract_fingerprint_{guide,atomic,filter}│
│  → build_claims_ir → repo_analysis → compile_task_spec  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     Phase 2 (v0.5)                      │
│          Claude Code Agent SDK as Primary Executor      │
│                                                         │
│  PLANNING ──▶ ENV_SETUP ──▶ EXECUTING ──▶ SUCCESS      │
│     ↑            │              │                       │
│     │            ▼              ▼                       │
│     └── REPLANNING ◀── failure analysis                 │
│              │                                          │
│              ▼ (attempts exhausted)                     │
│         AUTONOMOUS ──▶ SUCCESS / FAILED                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                        Phase 3                          │
│  Execution Evidence → Claim Verification → Verdict      │
│                                                         │
│  observe_metrics → align_evidence → verify_claims       │
│  → audit_report                                         │
└─────────────────────────────────────────────────────────┘
```

### Phase 2 执行模型 / Phase 2 Execution Model (v0.5)

**旧模型 (v0.4)**：先尝试 `env_mgr.run_in_env()` 直接执行，失败后触发 Codex CLI recovery。三层策略：direct → Codex recovery → autonomous。

**新模型 (v0.5)**：所有步骤统一通过 Claude Code Agent SDK 执行。

```python
_execute_step()
  → build_step_execution_prompt()   # 构建执行提示词
  → _run_claude()                   # Claude Code 执行
      → system_prompt: "conda run -n {env} ..."  # 环境指令
      → SDK query() → Bash tool 执行命令
      → 收集 stdout/stderr/metrics
  → _finalize_step_result()         # 提取指标、分类错误
```

**两级修复策略**：
- **Tier 1 (Micro-repair)**：行内修复（pip install、路径修正、设备切换），无需 LLM 重新规划
- **Tier 2 (Macro-replan)**：完整重新规划 + 环境重建周期

---

## 4. 数据模型 / Data Models

核心 Pydantic 模型定义于 `p2c/schemas.py`：

| 模型 | 用途 |
|---|---|
| `Fingerprint` | 论文指纹：配置 + claims + 证据锚点 |
| `FingerprintClaim` | 单条可验证 claim（含容差与逻辑） |
| `ClaimsIR` | 中间表示：实验 + claims + 推理 |
| `TaskSpec` | 可执行任务规格 + 入口点 + 指标观察器 |
| `MetricContract` | 指标解析契约（解析器 + 归一化规则） |
| `ExecutionPlan` | Phase 2 执行计划（环境依赖 + 执行步骤） |
| `ExecutionStep` | 原子执行步骤（命令、cwd、超时、依赖） |
| `Phase2State` | 状态机快照 |
| `VerdictDoc` | 最终结论：SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE |

---

## 5. 产物结构 / Artifact Layout

每次运行的产物存储在 `artifacts/<run_id>/` 下：

```
<run_id>/
├── fingerprint/
│   ├── fingerprint.json            # 最终指纹与 claims
│   ├── guide_sentences.json        # 筛选出的关键句/表格
│   ├── atomic_criteria.json        # 原子化标准
│   └── claims_ir.json              # Claims 中间表示
├── task/
│   ├── repo_analysis.json          # 仓库分析结果
│   ├── task_spec.json              # 可执行任务规格
│   └── metric_contract.json        # 指标解析契约
├── execution/
│   ├── execution_plan.json         # Phase 2 执行计划
│   ├── env_setup_result.json       # 环境验证结果
│   ├── phase2_state.json           # 状态机快照
│   ├── execution_failures.json     # 失败记录
│   ├── run.log                     # 完整执行日志
│   ├── env_lock/
│   │   └── pip_freeze.txt          # 环境锁定文件
│   └── codex_outputs/
│       ├── run_manifest.json       # 所有执行记录 + 指标
│       ├── claim_alignment.json    # Claim ↔ 指标对齐
│       └── step_*_{stdout,stderr}.log  # 每步日志
└── results/
    ├── metrics.json                # 解析后的指标
    ├── verdict.json                # 最终 claim 结论
    └── report.md                   # 人类可读审计报告
```

---

## 6. 安装 / Installation

### 依赖 / Dependencies

```bash
pip install pydantic>=2.0 openai>=1.0 claude-agent-sdk>=0.1.0
```

测试额外需要：

```bash
pip install pytest
```

### 环境要求 / Requirements

- Python 3.10+
- Conda（用于 Phase 2 目标仓库的环境隔离）

---

## 7. 环境变量 / Environment Variables

| 变量 | 用途 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | LLM 调用（Phase 1/3 的论文处理） | 必填 |
| `OPENAI_MODEL` | LLM 模型名 | `gpt-5.4` |
| `OPENAI_BASE_URL` | OpenAI 兼容 API 地址 | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Claude Code Agent SDK 调用（Phase 2 执行） | 必填 |
| `P2C_MAX_REPLAN` | Phase 2 最大重规划次数 | `3` |

Phase 2 执行时会自动转发以下宿主环境变量至 Claude Code：
`HOME`, `USER`, `PATH`, `LANG`, `SHELL`, `TERM`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `CONDA_EXE`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`

---

## 8. 使用方法 / How To Run

### Phase 1：论文摄取与任务编译

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md "/path/to/paper.md" \
  --paper_md_out "/path/to/output/paper.md" \
  --repo_dir "/path/to/Target" \
  --run_id my_audit \
  --artifacts_dir "/path/to/artifacts" \
  --budget_minutes 60
```

### Phase 2：Claude Code 执行

```bash
python -m p2c.main \
  --phase 2 \
  --paper_md "/path/to/output/paper.md" \
  --paper_md_out "/path/to/output/paper.md" \
  --repo_dir "/path/to/Target" \
  --run_id my_audit \
  --artifacts_dir "/path/to/artifacts" \
  --budget_minutes 60
```

### Phase 3：验证与报告

```bash
python -m p2c.main \
  --phase 3 \
  --paper_md "/path/to/output/paper.md" \
  --paper_md_out "/path/to/output/paper.md" \
  --repo_dir "/path/to/Target" \
  --run_id my_audit \
  --artifacts_dir "/path/to/artifacts" \
  --budget_minutes 60
```

### 全流程示例 / Full Pipeline Example

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Phase 1 → Phase 2 → Phase 3
for phase in 1 2 3; do
  python -m p2c.main \
    --phase $phase \
    --paper_md paper.md \
    --paper_md_out output/paper.md \
    --repo_dir ./Target \
    --run_id audit_001 \
    --budget_minutes 60
done

# 查看结果
cat artifacts/audit_001/results/verdict.json
cat artifacts/audit_001/results/report.md
```

---

## 9. 测试 / Testing

```bash
# 使用 uv（推荐，自动解决 Python 版本）
uv run --python 3.13 --with pytest --with pydantic -- \
  python -m pytest tests/test_phase2_fixes.py -v

# 或直接使用 pytest
python -m pytest tests/ -v
```

当前测试文件：

| 文件 | 覆盖范围 |
|---|---|
| `test_phase2_fixes.py` | Conda spec 构建、指标提取、环境转发、Claude Code 执行 mock |
| `test_phase1_repo_analysis.py` | 仓库分析 agent |
| `test_claim_context_pipeline.py` | 端到端 claim 构建流水线 |
| `test_table_extraction.py` | Markdown 表格解析 |

---

## 10. 失败分类体系 / Failure Taxonomy

Phase 2 使用 `failure_taxonomy.py` 中定义的分层失败分类：

| 层级 | 示例失败码 | 修复策略 |
|---|---|---|
| **Dependency** | `DEP_MISSING_PACKAGE`, `DEP_VERSION_CONFLICT` | INLINE_FIX / REPLAN |
| **Data** | `DATA_NOT_FOUND`, `DATASET_UNRESOLVED` | REPLAN / SKIP |
| **Configuration** | `CONFIG_INVALID_VALUE`, `CONFIG_DEPRECATED_OPTION` | INLINE_FIX |
| **Code** | `CODE_SYNTAX_ERROR`, `CODE_IMPORT_ERROR` | REPLAN |
| **Resource** | `RESOURCE_INSUFFICIENT_MEMORY`, `RESOURCE_GPU_UNAVAILABLE` | ABORT / SKIP |
| **Output** | `OUTPUT_MISSING_FILE`, `OUTPUT_PARSE_ERROR` | RETRY |

每个失败码包含：`repair_strategy`（修复策略）、`auto_repair_confidence`（自动修复置信度 0.0-1.0）、`is_fast_fail`（是否终止整条流水线）。

---

## 11. v0.5 迁移变更摘要 / v0.5 Migration Changelog

### 核心变更

| 项目 | v0.4 (Codex CLI) | v0.5 (Claude Code SDK) |
|---|---|---|
| 执行引擎 | OpenAI Codex CLI (`codex exec`) | Claude Code Agent SDK (`claude-agent-sdk`) |
| 执行模型 | 三层混合：direct → Codex recovery → autonomous | 统一走 Claude Code primary |
| 默认模型 | `gpt-5.4` | `claude-sonnet-4-20250514` |
| SDK 导入 | 懒加载 (lazy import) | 模块级导入 + try/except fallback |
| 环境传递 | `env_mgr.run_in_env()` subprocess | `system_prompt` 中指令 `conda run -n {env}` |
| 返回类型 | `subprocess.CompletedProcess` | `ClaudeResult` dataclass |

### 变更文件

- **`p2c/agents/phase2/codex_executor.py`** — 主执行器重写
- **`p2c/runtime/conda_env.py`** — 移除 Codex 相关逻辑
- **`p2c/agents/phase2/local_prompt_templates.py`** — 提示词更新
- **`requirements.txt`** — 新增 `claude-agent-sdk`
- **`tests/test_phase2_fixes.py`** — 10 个测试重写

### 删除的方法

`_run_codex`, `_prepare_command_for_execution`, `_split_assignments`, `_build_failure_context`, `_resolve_codex_bin`

---

## 12. 设计理念 / Design Philosophy

本项目的 Phase 2 执行模型参照 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的设计：

- **AI agent 即执行者**：不再需要 subprocess 中间层，Claude Code 直接通过 Bash tool 执行命令
- **Self-healing**：Claude Code 内置错误检测与自动修复能力，遇到失败会自主迭代
- **环境隔离**：通过 `conda run --no-capture-output -n {env_name}` 前缀实现，无需激活环境
- **最小编排**：宿主侧只负责构建 prompt、收集结果、提取指标，执行细节完全交给 agent

---

## 13. License

MIT
