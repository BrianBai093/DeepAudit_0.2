## Codex 开发提示词（当前系统实装版）

你是该仓库的工程实现代理。目标不是重写架构，而是在**现有代码基础上**稳定推进 `Paper2Code` 三阶段链路：

- Phase 1：论文处理与指纹抽取
- Phase 2：E2B Sandbox + Codex 执行
- Phase 3：指标对齐与双轨判定

请严格以仓库现状为准，不回退到旧版 mini-SWE 方案。

---

## 1. 当前系统真相（以代码为准）

### 1.1 编排入口

- CLI：`p2c/main.py`
- 编排：`p2c/graph.py`
- Agent 基类：`p2c/agents/base.py`
- LLM 封装：`p2c/llm/client.py`
- Artifact 管理：`p2c/io_artifacts.py`

### 1.2 Phase 流程（当前）

1. Phase 1（`run_phase_1`）
   - `ingest_paper`
   - `extract_fingerprint_guide`
   - `extract_fingerprint_atomic`
   - `extract_fingerprint_filter`
   - `build_claims_ir`
   - `compile_task_spec`
2. Phase 2（`run_phase_2`）
   - `prepare_sandbox`
   - `run_codex_exec`
   - `collect_codex_outputs`
3. Phase 3（`run_phase_3`）
   - `observe_metrics`
   - `align_evidence`
   - `verify_claims`
   - `audit_report`

---

## 2. 关键约束（必须遵守）

1. 不改 Phase 主流程节点顺序，除非用户明确要求。
2. 所有 Agent 必须继承 `BaseAgent`，通过 `safe_chat_text/safe_chat_json` 调 LLM（可降级，但不能抛弃该路径）。
3. 日志必须写控制台 + `execution/run.log`。
4. 所有结构化产物必须落到 `artifacts/{run_id}/...`，并尽量满足 schema。
5. 失败时必须保留可诊断信息，尤其 Phase 2 的 `execution/codex_failure.json`。
6. 禁止把当前系统描述成 mini-SWE 主执行链路；当前主执行是 E2B 内 `codex exec`。

---

## 3. Phase 1 设计基线

### 3.1 ingest_paper

- 输入：`--paper_md`（通常 `Target/paper/full.md`）
- 输出：`--paper_md_out`（通常 `output/paper.md`）
- 行为：调用 `p2c/agents/PictureToWords.py` 把 markdown 图片替换为文字描述。

### 3.2 指纹三阶段

1. `extract_fingerprint_guide`
   - 基于 `output/paper.md` 召回可执行复现单元（句子 + table_block）
   - 产物：`fingerprint/guide_sentences.json`
2. `extract_fingerprint_atomic`
   - 原子化 `<fact>/<scope>`，表格指标展开
   - 产物：`fingerprint/atomic_criteria.json`、`fingerprint/atomic_rejected.json`
3. `extract_fingerprint_filter`
   - 去重筛选 + 组装最终 `fingerprint.json`
   - 中间产物：`fingerprint/filter_clusters.json`、`fingerprint/filter_selected.json`

### 3.3 下游任务编译

- `build_claims_ir`：优先消费 `fingerprint.claims` 生成 `fingerprint/claims_ir.json`
- `compile_task_spec`：扫描 repo 入口并生成
  - `task/task_spec.json`
  - `task/metric_contract.json`

---

## 4. Phase 2 设计基线（E2B + Codex）

### 4.1 Runtime

- 运行时工厂：`p2c/runtime/factory.py`
- 默认后端：`P2C_RUNTIME_BACKEND=e2b`
- E2B 实现：`p2c/runtime/e2b_runtime.py`
- 强制模板：`openai-codex`

### 4.2 prepare_sandbox

在 sandbox 中准备：

- `workspace_root`（自动探测可写路径）
- `{root}/repo`（上传本地 `--repo_dir`）
- `{root}/data`（自动扫描数据目录/文件并映射）
- `{root}/inputs/task_spec.json`
- `{root}/inputs/claims_ir.json`
- `{root}/outputs`

默认排除 `.git` 上传；设置 `P2C_INCLUDE_GIT=1` 可包含。

### 4.3 run_codex_exec（当前固定策略）

Codex 命令固定追加参数：

- `--skip-git-repo-check`
- `--dangerously-bypass-approvals-and-sandbox`

默认模型：

- `gpt-5.1-codex-mini`（可用 `P2C_CODEX_MODEL` 覆盖）

执行方式：

- 后台启动 + 轮询 + 实时日志增量输出
- 主执行后做输出校验；必要时进入 repair-only
- 若全部 entrypoint 因依赖不可运行，抛 `DEPENDENCY_UNRESOLVED`

关键输出：

- `execution/codex_outputs/run_manifest.json`
- `execution/codex_outputs/claim_alignment.json`
- `execution/codex_outputs/codex_worklog.jsonl`
- `execution/codex_outputs/patches.diff`
- `execution/codex_outputs/codex_exec.log`
- 可选：`dependency_solver.json`、`pip_install.log`
- 失败诊断：`execution/codex_failure.json`

### 4.4 collect_codex_outputs

- 从 sandbox 拉回上述输出到 `execution/codex_outputs/`
- 严格校验 `run_manifest.json` 与 `claim_alignment.json`
- 生成 `execution/repo_state.json`（gitless 允许，`NO_GIT_METADATA`）

---

## 5. Phase 3 设计基线（双轨）

1. `observe_metrics`
   - 读取 `execution/codex_outputs/run_manifest.json`
   - 输出 `results/metrics.json`
2. `align_evidence`
   - 输入：`claims_ir + claim_alignment + metrics`
   - 输出：
     - `results/parsed_evidence.json`（数值证据轨）
     - `results/evaluability.json`（可评估性轨）
3. `verify_claims`
   - 输出：
     - `results/verdict.json`（SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE）
     - `results/evaluability_verdict.json`（EVALUABLE/PARTIAL/NOT_EVALUABLE）
4. `audit_report`
   - 输出 `results/report.md`
   - 读取并汇总 Phase 1-3 artifacts，gitless 场景显示 `N/A (gitless run)`

---

## 6. CLI 与阶段前置条件

### 6.1 统一命令

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md Target/paper/full.md \
  --paper_md_out output/paper.md \
  --repo_dir Target/code \
  --run_id demo_run_001 \
  --artifacts_dir artifacts \
  --budget_minutes 30 \
  --max_self_heal_iters 2
```

### 6.2 前置校验

- Phase 2 需要 Phase 1 的有效 `task/task_spec.json`（且 `entrypoints` 非空）
- Phase 3 需要 Phase 2 的
  - `execution/codex_outputs/run_manifest.json`
  - `execution/codex_outputs/claim_alignment.json`

---

## 7. 环境变量（当前生效）

- `OPENAI_API_KEY`（必需）
- `E2B_API_KEY`（Phase 2 必需）
- `P2C_RUNTIME_BACKEND`（默认 `e2b`，可设 `local`）
- `P2C_SANDBOX_TIMEOUT_SEC`（默认 3600）
- `P2C_CODEX_MODEL`（默认 `gpt-5.1-codex-mini`）
- `P2C_INCLUDE_GIT`（默认不上传 `.git`）
- `P2C_WORKSPACE_ROOT`（可选覆盖 workspace 根目录）

---

## 8. 产物规范（当前 required files）

以 `p2c/io_artifacts.py` 的 `REQUIRED_FILES` 为真源。核心文件包含：

- `fingerprint/*.json`（guide/atomic/filter/final/claims_ir）
- `task/task_spec.json`、`task/metric_contract.json`
- `execution/run.log`
- `execution/codex_outputs/*`（run_manifest/claim_alignment/worklog/log/diff 等）
- `execution/codex_failure.json`
- `results/metrics.json`
- `results/parsed_evidence.json`
- `results/evaluability.json`
- `results/evaluability_verdict.json`
- `results/verdict.json`
- `results/report.md`

---

## 9. 开发/改动原则

1. 优先修正真实阻塞问题，不做大而空重构。
2. 新逻辑必须写 reason_codes，确保失败可审计。
3. 对外契约变更前先同步 `schemas.py + io_artifacts.py + tests`。
4. 若用户要求“按当前系统更新”，必须先核对 `graph.py/main.py/io_artifacts.py`，再改文档或实现。

