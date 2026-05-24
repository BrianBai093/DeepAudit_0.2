# DeepAudit v0.5

> 论文可复现性自动审计系统 / Automated Paper Reproducibility Audit System

DeepAudit takes a research paper and its companion code repository, then runs a three-phase audit to determine whether the paper's executable claims are reproducible from the available code and evidence.

## 1. 项目简介 / What This Repo Does

DeepAudit 的输入是：

- 一份论文 Markdown，通常来自 PDF 转换后的 `paper.md`
- 可选的论文 PDF，用于图表视觉抽取
- 一个待审计目标代码仓库，例如 `Target/code`

DeepAudit 的输出是：

- 论文 claim / experiment / metric 的结构化中间产物
- 目标仓库环境构建与执行日志
- Phase 2 执行证据包
- Phase 3 claim verdict、可复现性评分和人类可读报告

流水线分三阶段：

1. **Phase 1: Paper + Repo Understanding**  
   读取论文，抽取可验证 claim，解析图表/表格，分析目标仓库依赖和候选入口点，生成 `claims_ir.json`、`task_spec.json` 和 `metric_contract.json`。

2. **Phase 2: Environment + Autonomous Execution**  
   根据仓库依赖创建隔离 conda/venv 环境，然后用 Claude Code Agent SDK 在目标仓库中自主执行实验。执行器会记录命令、日志、指标、失败原因，并生成 Phase 3 消费的规范证据包。

3. **Phase 3: Evidence Alignment + Verdict**  
   从 Phase 2 证据包提取指标，对齐到论文 claim，判断是否支持论文结论，并生成 `verdict.json`、`reproducibility_score.json` 和 `report.md`。

当前 v0.5 的关键点：Phase 2 的真实实现是 `ToolAgent + ExecutorAgent + Phase2Orchestrator`。README 中的 Phase 2 描述已经按当前代码路径和产物格式校准。

## 2. 目录结构 / Repository Layout

```text
DeepAudit_0.2/
├── p2c/
│   ├── main.py                         # CLI 入口
│   ├── graph.py                        # 三阶段 agent 编排
│   ├── schemas.py                      # Pydantic 数据模型
│   ├── io_artifacts.py                 # artifacts/<run_id>/ 产物树管理
│   ├── failure_taxonomy.py             # Phase 2 失败分类
│   ├── agents/
│   │   ├── base.py                     # BaseAgent
│   │   ├── phase1/
│   │   │   ├── ingest_paper.py
│   │   │   ├── extract_visual_elements.py
│   │   │   ├── extract_fingerprint_guide.py
│   │   │   ├── extract_fingerprint_atomic.py
│   │   │   ├── enrich_claims_visual.py
│   │   │   ├── extract_fingerprint_filter.py
│   │   │   ├── build_claims_ir.py
│   │   │   ├── repo_analysis.py
│   │   │   └── compile_task_spec.py
│   │   ├── phase2/
│   │   │   ├── tool_agent.py           # 环境规格生成与安装
│   │   │   ├── executor_agent.py       # Claude Code 自主执行器
│   │   │   ├── orchestrator.py         # env setup / repair / execute 状态机
│   │   │   └── result_extraction.py    # 指标抽取与失败分类
│   │   └── phase3/
│   │       ├── execution_summary_evidence.py
│   │       ├── observe_metrics.py
│   │       ├── align_evidence.py
│   │       ├── verify_claims.py
│   │       ├── score_and_diagnose.py
│   │       ├── visual_to_repo_alignment.py
│   │       ├── execution_log_evidence.py
│   │       ├── reproduce_figures.py
│   │       └── audit_report.py
│   ├── llm/client.py                   # OpenAI-compatible LLM client
│   ├── rag/                            # 可选代码索引 / RAG
│   ├── runtime/conda_env.py            # conda/venv 环境管理
│   └── utils/
├── scripts/run_audit.sh                # 一键运行脚本
├── tests/                              # 回归测试
├── Target/                             # 待审计目标材料
├── artifacts/                          # 运行产物
├── output/                             # 论文 markdown 中间输出
└── requirements.txt
```

## 3. 架构概览 / Pipeline

```text
Phase 1
paper.md (+ optional paper.pdf) + repo
  -> PaperText
  -> visual_elements / visual_targets
  -> fingerprint / atomic criteria
  -> claims_ir
  -> repo_analysis / task_spec / metric_contract

Phase 2
repo_analysis + claims_ir + metric_contract
  -> executor_env_spec
  -> env_setup_result
  -> Claude executor session
  -> executor_results / run_manifest
  -> phase2_execution_package

Phase 3
phase2_execution_package + claims_ir + visual_targets
  -> effective evidence
  -> metrics
  -> claim alignment
  -> verdict / score / report
```

`p2c/graph.py` 是实际三阶段调用顺序的来源。`p2c/main.py` 负责 CLI 参数、阶段前置产物校验、全局日志和 `execution/context.json` 写入。

## 4. Phase 2 当前执行模型

Phase 2 不再使用旧的 plan/replan 文件模型。当前执行链路如下：

1. `ToolAgent.build_env_spec()`  
   读取 `task/repo_analysis.json`，从 `requirements.txt`、`pyproject.toml`、`environment.yml`、editable install 等信息推导 `ExecutorEnvSpec`，写入 `execution/executor_env_spec.json`。

2. `ToolAgent.run()`  
   创建隔离环境，安装 conda/pip/system/pre-install 依赖，校验关键 import，并写入 `execution/env_setup_result.json` 和 `execution/env_lock/pip_freeze.txt`。

3. `ExecutorAgent.run()`  
   用 Claude Code Agent SDK 开启一次自主执行 session。它把 repo README、依赖文件、实验列表、metric contract 和明确的输出目录传给 Claude executor，然后要求其在 `execution/executor_outputs/` 写出标准执行结果。
   在训练开始前，宿主进程会扫描目标仓库中常见的保存逻辑，例如 `np.savetxt`、`torch.save`、`plt.savefig`、`json.dump`、`savepath` / `output_dir` 等，写入 `execution/phase2_artifacts/artifact_storage_preflight.json`，并把摘要交给 executor，帮助它选择正确 cwd 和 repo 支持的输出参数。
   在训练结束后，宿主进程会再次扫描目标仓库，复制新生成或更新的结果数据、checkpoint 和图片到 `execution/phase2_artifacts/files/`，并写入 `execution/phase2_artifacts/manifest.json`。这些文件会并入 `run_manifest.json` 和 `phase2_execution_package.json`，因此 Phase 3 不再只依赖 stdout。

4. `Phase2Orchestrator`  
   控制 env setup -> executing -> repairing/success/failed。遇到依赖类失败时会做有限环境 patch，例如补装缺失包、处理 CUDA/device 问题。

Phase 2 的规范输出是：

- `execution/executor_outputs/phase2_execution_package.json`
- `execution/executor_outputs/PHASE2_RESULTS.md`
- `execution/executor_outputs/run_manifest.json`
- `execution/executor_outputs/executor_results.json`
- `execution/executor_outputs/executor_activity.jsonl`
- `execution/executor_outputs/session_stdout.log`
- `execution/executor_outputs/session_stderr.log`
- `execution/executor_outputs/executor_agent.log`
- `execution/executor_outputs/executor_runtime.json`
- `execution/phase2_artifacts/artifact_storage_preflight.json`
- `execution/phase2_artifacts/manifest.json`

Phase 3 优先消费 `phase2_execution_package.json`。`run_manifest.json` 和 `executor_results.json` 主要作为兼容与诊断输入。

## 5. 核心数据模型 / Core Data Models

主要模型都在 `p2c/schemas.py`：

| 模型 | 用途 |
|---|---|
| `PaperText` | 论文文本、章节、图表描述 |
| `VisualElement` / `VisualTarget` | 从 PDF 图表中抽取的视觉证据和待复现图表目标 |
| `Fingerprint` / `FingerprintClaim` | 论文指纹与初始 claim |
| `AtomicCriterion` | 从论文句子/表格/视觉信息抽取的原子验证标准 |
| `ClaimsIR` / `Experiment` / `ClaimItem` | Phase 1 到 Phase 2/3 的 claim 中间表示 |
| `RepoAnalysis` / `DependencyProfile` | 目标仓库生态、依赖文件和入口点候选 |
| `TaskSpec` / `MetricContract` | 可执行任务规格和指标解析契约 |
| `ExecutorEnvSpec` / `EnvSetupResult` | Phase 2 环境规格和安装结果 |
| `ExecutionRun` / `RunManifestDoc` | Phase 2 执行记录 |
| `MetricsDoc` / `ParsedEvidence` | Phase 3 指标和 claim 证据对齐 |
| `VerdictDoc` / `ReproducibilityScore` | 最终结论和评分 |

`ExecutionPlan` 仍保留在 schema 中，但注释明确为 deprecated compatibility model；当前 Phase 2 主路径不依赖它。

## 6. 产物结构 / Artifact Layout

每次运行写入 `artifacts/<run_id>/`：

```text
artifacts/<run_id>/
├── fingerprint/
│   ├── fingerprint.json
│   ├── guide_sentences.json
│   ├── atomic_criteria.json
│   ├── atomic_rejected.json
│   ├── filter_clusters.json
│   ├── filter_selected.json
│   ├── claims_ir.json
│   ├── visual_elements.json
│   └── visual_targets.json
├── task/
│   ├── repo_analysis.json
│   ├── task_spec.json
│   └── metric_contract.json
├── execution/
│   ├── context.json
│   ├── run.log
│   ├── executor_env_spec.json
│   ├── env_setup_result.json
│   ├── execution_failures.json
│   ├── phase2_state.json
│   ├── env_lock/
│   │   └── pip_freeze.txt
│   ├── executor_outputs/
│   │   ├── phase2_execution_package.json
│   │   ├── PHASE2_RESULTS.md
│   │   ├── run_manifest.json
│   │   ├── executor_results.json
│   │   ├── executor_activity.jsonl
│   │   ├── executor_runtime.json
│   │   ├── executor_agent.log
│   │   ├── session_stdout.log
│   │   ├── session_stderr.log
│   │   └── experiment_*_{stdout,stderr,narrative}.log
│   └── phase2_artifacts/
│       ├── artifact_storage_preflight.json
│       ├── manifest.json
│       └── files/
│           └── <repo-relative generated outputs>
└── results/
    ├── execution_summary_evidence.json
    ├── execution_log_evidence.json
    ├── effective_run_manifest.json
    ├── effective_claims_ir.json
    ├── metrics.json
    ├── parsed_evidence.json
    ├── evaluability.json
    ├── evaluability_verdict.json
    ├── verdict.json
    ├── reproducibility_score.json
    ├── visual_to_repo_alignment.json
    ├── reproduced_figures.json
    └── report.md
```

`ArtifactManager.ensure_tree()` 会先创建占位 JSON，因此某个文件存在不代表该阶段已经真正完成。判断 Phase 2 是否有效时，应看 `phase2_execution_package.json` 或 `run_manifest.json` 中是否有实验/运行记录。

## 7. 安装 / Installation

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

核心依赖见 `requirements.txt`：

- `pydantic`
- `openai`
- `claude-agent-sdk`
- `numpy`
- `PyMuPDF`
- `matplotlib`

Phase 2 会优先使用 conda 创建目标执行环境；如果 conda 不可用，`p2c/runtime/conda_env.py` 中也有 venv fallback。

## 8. 环境变量 / Environment Variables

| 变量 | 用途 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | Phase 1/3 的 OpenAI-compatible LLM 调用 | 必填 |
| `OPENAI_MODEL` | Phase 1/3 文本与视觉模型 | `gpt-5.4` |
| `OPENAI_BASE_URL` | OpenAI-compatible API 地址 | `https://api.openai.com/v1` |
| `OPENAI_TIMEOUT_SEC` | 文本/JSON 调用超时 | `300` |
| `OPENAI_VISION_TIMEOUT_SEC` | 视觉调用超时 | `max(360, OPENAI_TIMEOUT_SEC)` |
| `MINERU_API_TOKEN` | MinerU 精准解析 API Token；当 PDF 超过轻量接口限制时需要 | 未设置 |
| `P2C_MINERU_MODE` | PDF→Markdown 转换模式：`auto` / `agent` / `standard` / `off` | `auto` |
| `P2C_MINERU_LANGUAGE` | MinerU 解析语言 | `en` |
| `P2C_MINERU_TIMEOUT_SEC` | MinerU 任务轮询超时 | `900` |
| `ANTHROPIC_API_KEY` | Claude Code Agent SDK | Phase 2 必填 |
| `P2C_CLAUDE_MODEL` | Claude executor 模型 | `claude-haiku-4-5-20251001` |
| `P2C_MAX_ENV_PATCH` | Phase 2 环境 patch 最大尝试次数 | `2` |
| `P2C_LAYERED_INSTALL` | 是否分层安装依赖 | `1` |
| `P2C_KEEP_CONDA_ENV` | 设置后保留 Phase 2 环境，便于调试 | 未设置 |
| `P2C_HOST_TOOL_DIRS` | 额外转发给执行环境的宿主工具路径 | 未设置 |
| `P2C_VENV_ROOT` | venv fallback 根目录 | `/tmp` |
| `P2C_PHASE2_ARTIFACT_MAX_MB` | Phase 2 自动收集单个训练产物的大小上限 | `50` |

`scripts/run_audit.sh` 还会读取并转发一些运行预算变量，例如 `BUDGET_MINUTES`、`P2C_MIN_EXEC_TIMEOUT_SEC`、`P2C_ATOMIC_LLM_SENTENCE_BUDGET`、`P2C_ATOMIC_LLM_TABLE_BUDGET`。

## 9. 使用方法 / How To Run

### 单阶段运行

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md_out output/paper.md \
  --paper_pdf Target/paper.pdf \
  --repo_dir Target/code \
  --run_id audit_001 \
  --artifacts_dir artifacts \
  --budget_minutes 60
```

```bash
python -m p2c.main \
  --phase 2 \
  --paper_md output/paper.md \
  --paper_md_out output/paper.md \
  --repo_dir Target/code \
  --run_id audit_001 \
  --artifacts_dir artifacts \
  --budget_minutes 180
```

```bash
python -m p2c.main \
  --phase 3 \
  --paper_md output/paper.md \
  --paper_md_out output/paper.md \
  --repo_dir Target/code \
  --run_id audit_001 \
  --artifacts_dir artifacts \
  --budget_minutes 60
```

### 全流程运行

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

for phase in 1 2 3; do
  PDF_ARGS=()
  if [ "$phase" = "1" ]; then
    PDF_ARGS=(--paper_pdf Target/paper.pdf)
  fi
  python -m p2c.main \
    --phase "$phase" \
    --paper_md Target/paper/full.md \
    --paper_md_out output/paper.md \
    --repo_dir Target/code \
    --run_id audit_001 \
    --artifacts_dir artifacts \
    --budget_minutes 180 \
    "${PDF_ARGS[@]}"
done
```

Phase 1 才需要 `--paper_pdf`；如果没有 PDF，图表抽取和视觉增强会跳过，文本 claim 流水线仍可运行。
当 `--paper_md` 省略且提供了 `--paper_pdf` 时，Phase 1 会自动将 `Target/paper.pdf` 转成 `Target/paper/full.md`，再生成 `--paper_md_out`。如果 `full.md` 已存在且不旧于 PDF，会直接复用；超过 MinerU 轻量接口限制的 PDF 需要设置 `MINERU_API_TOKEN`。

### 脚本运行

先编辑 `scripts/run_audit.sh` 顶部 DEFAULTS，使路径匹配本机，然后运行：

```bash
./scripts/run_audit.sh audit_001
./scripts/run_audit.sh audit_001 1,2
./scripts/run_audit.sh audit_001 3
```

结果查看：

```bash
cat artifacts/audit_001/results/verdict.json
cat artifacts/audit_001/results/report.md
cat artifacts/audit_001/execution/executor_outputs/PHASE2_RESULTS.md
```

## 10. 测试 / Testing

```bash
python -m pytest tests/ -v
```

常用定向测试：

```bash
python -m pytest tests/test_main_context.py -v
python -m pytest tests/test_phase2_fixes.py -v
python -m pytest tests/test_phase3_report.py -v
python -m pytest tests/test_visual_enrichment.py -v
```

测试覆盖重点包括：

- Phase 1 repo analysis、table extraction、claim context 和 visual enrichment
- Phase 2 env spec、executor outputs、phase2 execution package、failure classification
- Phase 3 metrics observation、effective evidence、claim verification、visual alignment、figure reproduction 和 report rendering

## 11. 失败分类 / Failure Taxonomy

Phase 2 通过 `failure_taxonomy.py` 和 `result_extraction.py` 对失败进行分类。常见类别：

| 类别 | 示例 | 处理方式 |
|---|---|---|
| Dependency | 缺包、版本冲突、build failure | pip/conda patch 或失败记录 |
| Data | 数据集缺失、路径找不到 | 记录 blocker / 可能跳过 |
| Configuration | 参数过期、device 配置错误 | inline fix / CPU fallback |
| Code | import/runtime/syntax 错误 | 记录失败与日志 |
| Resource | GPU 不可用、内存不足、超时 | budget/guardrail 处理 |
| Output | executor 未写结果、指标无法解析 | 诊断为证据不足 |

每个 `StepFailure` 可包含 `failure_code`、`failure_layer`、`repair_strategy`、`repair_action` 和 `auto_repair_confidence`，便于 Phase 2 patch 和 Phase 3 报告解释。

## 12. 当前实现注意事项 / Implementation Notes

- `execution/executor_outputs/phase2_execution_package.json` 是 Phase 3 的首选执行证据来源。
- `execution/run.log` 是宿主 orchestration 日志；Claude session 的文本输出分别在 `executor_agent.log`、`session_stdout.log`、`session_stderr.log`。
- `ExecutorAgent` 有 repo mutation guard。若 Claude executor 修改了目标仓库中已跟踪源码，Phase 2 会拒绝该次运行并记录 `SOURCE_MUTATION_DETECTED`。
- `ArtifactManager.ensure_tree()` 会生成占位文件。调试时应检查文件内容中的 `reason_codes`，不要只看文件是否存在。
- `scripts/run_audit.sh` 中的 `PROJECT_ROOT`、`PAPER_MD`、`PAPER_PDF`、`REPO_DIR` 是本地路径默认值，需要按机器调整。

## 13. License

MIT
