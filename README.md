# DeepAudit v0.4

**Automated ML Paper Reproducibility Verification System**

**ML 论文可复现性自动验证系统**

---

## Overview | 概述

DeepAudit is a 3-phase pipeline that automatically verifies whether a code repository can reproduce the results claimed in its companion research paper. It ingests the paper, extracts quantitative claims, executes the repository code in an isolated environment, and produces a structured reproducibility audit report with a 0–10 score.

DeepAudit 是一个三阶段流水线系统，自动验证代码仓库是否能复现对应论文中声明的实验结果。它解析论文、提取定量声明、在隔离环境中执行仓库代码，最终生成结构化的可复现性审计报告（0-10 分评分）。

```
Paper (Markdown) ──┐
                   ├──▶ Phase 1: Extract Claims ──▶ Phase 2: Execute Code ──▶ Phase 3: Verify & Report
Repository ────────┘
```

---

## Architecture | 架构

### Phase 1 — Paper Ingestion & Claim Extraction | 论文解析与声明提取

Converts the paper into structured, verifiable claims grouped by experiment.

将论文转化为按实验分组的结构化、可验证声明。

| Step | Agent | Description |
|------|-------|-------------|
| 1 | `IngestPaperAgent` | Converts markdown images to text |
| 2 | `ExtractFingerprintGuideAgent` | Splits paper into sentences and table blocks |
| 3 | `ExtractFingerprintAtomicAgent` | Extracts atomic criteria via LLM |
| 4 | `ExtractFingerprintFilterAgent` | Filters and clusters atomic criteria |
| 5 | `BuildClaimsIRAgent` | **LLM-driven**: identifies experiments, groups claims, assesses repo coverage |
| 6 | `RepoAnalysisAgent` | Analyzes repo structure (entrypoints, dependencies) |
| 7 | `CompileTaskSpecAgent` | Compiles executable task specification |

**Key outputs | 关键产物:**
- `fingerprint/claims_ir.json` — Experiments + claims with `repo_coverage` assessment
- `task/repo_analysis.json` — Repository entrypoints and dependency profiles
- `task/metric_contract.json` — Regex patterns for metric extraction

### Phase 2 — Local Execution | 本地执行

Executes the repository code in an isolated conda/venv environment.

在隔离的 conda/venv 环境中执行仓库代码。

| Step | Agent | Description |
|------|-------|-------------|
| 8 | `PlannerAgent` | LLM generates `ExecutionPlan` (deps, steps, expected results) |
| 9 | `ToolAgent` | Creates conda env, layered dependency install (core → ML → paper-specific) |
| 10 | `CodexExecutorAgent` | **Hybrid execution**: direct subprocess + Codex recovery fallback |
| 11 | `Phase2Orchestrator` | Plan-Execute-ReAct loop with two-tier repair (micro/macro) |

**Execution strategy | 执行策略:**
1. **Direct execution** — Commands run via `env_mgr.run_in_env()` (guaranteed correct environment)
2. **Codex recovery** — If direct fails, Codex CLI diagnoses and fixes
3. **Autonomous exploration** — Full Codex agent fallback when all plans fail

**Key outputs | 关键产物:**
- `execution/codex_outputs/run_manifest.json` — All runs with extracted metrics
- `execution/codex_outputs/claim_alignment.json` — Claims mapped to metrics
- `execution/phase2_state.json` — Orchestrator state

### Phase 3 — Verification & Reporting | 验证与报告

Compares execution results against paper claims and produces the audit report.

将执行结果与论文声明进行比对，生成审计报告。

| Step | Agent | Description |
|------|-------|-------------|
| 12 | `ObserveMetricsAgent` | Extracts metrics from run manifest + stdout logs |
| 13 | `AlignEvidenceAgent` | Aligns claims to metrics; gates by `repo_coverage` |
| 14 | `VerifyClaimsAgent` | LLM-driven verdicts: SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE |
| 15 | `AuditReportAgent` | LLM generates full report with 0-10 reproducibility score |

**Verdict logic | 判定逻辑:**
- `repo_coverage = "not_found"` → claim is `INCONCLUSIVE` (experiment not in repo)
- Config claims → `INCONCLUSIVE` (need code evidence, not metric matching)
- Result claims with matched metrics → tolerance check: `|reproduced - target| ≤ max(abs_eps, rel_eps × |target|)`

**Key outputs | 关键产物:**
- `results/verdict.json` — Per-claim verdicts with `experiments_summary`
- `results/report.md` — Full audit report (Executive Summary, Scoring Breakdown, Gaps)

---

## Usage | 使用方法

### Prerequisites | 前置要求

- Python 3.10+
- conda or miniconda (for Phase 2 environment isolation)
- OpenAI API key (for LLM calls)
- Codex CLI (optional, for Phase 2 recovery mode)

### Run | 运行

```bash
# Set API key | 设置 API Key
export OPENAI_API_KEY="sk-..."

# Phase 1: Extract claims from paper | 从论文提取声明
python -m p2c.main --phase 1 \
  --paper_md paper.md \
  --paper_md_out paper_processed.md \
  --repo_dir ./target_repo \
  --run_id my_audit_001

# Phase 2: Execute repository code | 执行仓库代码
python -m p2c.main --phase 2 \
  --paper_md paper.md \
  --paper_md_out paper_processed.md \
  --repo_dir ./target_repo \
  --run_id my_audit_001 \
  --budget_minutes 30

# Phase 3: Verify and generate report | 验证并生成报告
python -m p2c.main --phase 3 \
  --paper_md paper.md \
  --paper_md_out paper_processed.md \
  --repo_dir ./target_repo \
  --run_id my_audit_001
```

### CLI Arguments | 命令行参数

| Argument | Required | Description |
|----------|----------|-------------|
| `--phase` | Yes | Phase to run: `1`, `2`, or `3` |
| `--paper_md` | Yes | Input paper markdown path |
| `--paper_md_out` | Yes | Output processed paper path |
| `--repo_dir` | Yes | Path to repository under audit |
| `--run_id` | Yes | Unique run identifier |
| `--artifacts_dir` | No | Artifacts directory (default: `./artifacts`) |
| `--budget_minutes` | No | Phase 2 time budget (default: `30`) |
| `--max_self_heal_iters` | No | Max replanning iterations (default: `6`) |

---

## Environment Variables | 环境变量

### LLM Configuration | LLM 配置

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `OPENAI_MODEL` | `gpt-5.4` | LLM model for all agents |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |

### Phase 2 Execution | Phase 2 执行配置

| Variable | Default | Description |
|----------|---------|-------------|
| `P2C_CODEX_BIN` | auto-detect | Path to Codex CLI binary |
| `P2C_CODEX_MODEL` | `gpt-5.4` | Model for Codex recovery |
| `P2C_MAX_REPLAN` | `3` | Max replan attempts |
| `P2C_LAYERED_INSTALL` | `1` | Enable layered dependency install |
| `P2C_KEEP_CONDA_ENV` | — | Set to keep env after run |
| `P2C_VENV_ROOT` | `/tmp` | Root for venv fallback |

---

## Artifact Structure | 产物结构

```
artifacts/<run_id>/
├── fingerprint/
│   ├── fingerprint.json          # Raw paper fingerprint
│   ├── guide_sentences.json      # Sentence-level extraction
│   ├── atomic_criteria.json      # Atomic claims from LLM
│   ├── filter_selected.json      # Filtered claims
│   └── claims_ir.json            # ★ Experiments + claims (Phase 1 final)
├── task/
│   ├── repo_analysis.json        # Repo entrypoints & deps
│   ├── task_spec.json            # Task specification
│   └── metric_contract.json      # Metric extraction regexes
├── execution/
│   ├── execution_plan.json       # LLM-generated execution plan
│   ├── env_setup_result.json     # Environment setup status
│   ├── phase2_state.json         # Orchestrator state
│   ├── env_lock/pip_freeze.txt   # Installed packages snapshot
│   └── codex_outputs/
│       ├── run_manifest.json     # ★ All runs with metrics (Phase 2 final)
│       ├── claim_alignment.json  # Claims → metrics mapping
│       └── step_*_stdout.log     # Per-step execution logs
└── results/
    ├── metrics.json              # Extracted metrics
    ├── parsed_evidence.json      # Claims matched to evidence
    ├── evaluability.json         # Evaluability assessment
    ├── verdict.json              # ★ Per-claim verdicts (Phase 3 final)
    └── report.md                 # ★ Full audit report with score
```

---

## Key Design Decisions | 关键设计决策

### Experiment-level claim grouping | 实验级声明分组

Claims are grouped by distinct experiments (e.g., "balanced 1:1 evaluation" vs "grouped imbalance evaluation"). Each experiment has a `repo_coverage` field (`implemented` / `partial` / `not_found`) that gates Phase 3 evaluation — claims from unimplemented experiments are marked `INCONCLUSIVE`, not compared against unrelated metrics.

声明按不同实验分组（如"均衡 1:1 评估"与"分组不平衡评估"）。每个实验有 `repo_coverage` 字段，Phase 3 据此决策：未实现实验的声明标记为 `INCONCLUSIVE`，不与无关指标比对。

### Unique metric naming | 唯一指标命名

The LLM is instructed to generate specific, unique metric names (e.g., `class_1_precision`, `weighted_f1`) rather than bare names like `precision`. This prevents ambiguity when multiple claims reference the same metric type across different experiments or classes.

LLM 被要求生成具体、唯一的指标名（如 `class_1_precision`、`weighted_f1`），而非裸名 `precision`，避免跨实验/类别的指标混淆。

### Hybrid execution (Direct + Codex) | 混合执行

Phase 2 first runs commands directly via `conda run` (guaranteeing correct environment), then falls back to Codex CLI only when direct execution fails. This avoids the environment isolation issue where Codex's internal shell doesn't inherit the conda env's PATH.

Phase 2 先通过 `conda run` 直接执行命令（保证环境正确），仅当直接执行失败时才回退到 Codex CLI，避免 Codex 内部 shell 无法继承 conda 环境 PATH 的隔离问题。

### Setup steps skip metric extraction | Setup 步骤跳过指标提取

Steps marked `is_setup=True` (e.g., source code inspection, data validation) do not undergo metric extraction. This prevents false-positive regex matches on source code literals like `"precision": 0.4`.

标记为 `is_setup=True` 的步骤（如源码查看、数据验证）不进行指标提取，防止对源码中 `"precision": 0.4` 等字面量的正则误匹配。

---

## Report Scoring | 报告评分

The final audit report scores reproducibility on 5 dimensions (0–2 points each, 10 total):

最终审计报告从 5 个维度评分（每项 0-2 分，总分 10 分）：

| Dimension | Description |
|-----------|-------------|
| **Code Completeness** | Does the repo implement all paper experiments? |
| **Execution Success** | Did all steps run without errors? |
| **Result Accuracy** | Do reproduced metrics match paper claims? |
| **Documentation** | Is the paper-to-code mapping clear? |
| **Data Availability** | Is training/test data accessible? |

---

## Tests | 测试

```bash
python -m pytest tests/ -v
```

| Test File | Coverage |
|-----------|----------|
| `test_claim_context_pipeline.py` | Claim grouping, alignment, disambiguation, verdict |
| `test_phase1_repo_analysis.py` | Entrypoint detection, notebook handling |
| `test_phase2_fixes.py` | Env setup, metric extraction, Codex binary resolution |
| `test_table_extraction.py` | Table parsing, plausibility checks |

---

## License | 许可证

This project is for research purposes.

本项目仅供研究用途。
