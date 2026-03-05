# Paper2Code (Dev)

## 中文

### 当前状态

本仓库当前采用三阶段架构：

1. `Phase1`：从论文和代码中抽取可执行任务，生成 `task_spec.json`（仅执行任务，不再写 `goal=[claim_ids]`）。
2. `Phase2`：在 E2B 沙盒中执行任务（Gitless），由 Runner 本地汇总结构化产物。
3. `Phase3`：本地做 claims 证据对齐与判定。

核心变化：

- 已放弃 SWE-agent 执行链路，统一使用 `E2B + codex exec`。
- Phase2 中 Codex 只负责执行任务，不再在沙盒里做 claims 判断。
- `claim_alignment.json` 改为 Runner 本地生成（路径不变）。

### Agent 目录

- `p2c/agents/phase1/*`
- `p2c/agents/phase2/*`
- `p2c/agents/phase3/*`

编排入口：`p2c/graph.py`，主入口：`p2c/main.py`。

### Phase2 关键行为（当前实现）

1. **Gitless 执行**
   - 本地 `--repo_dir` 上传到沙盒后直接执行。
   - Codex 固定参数：
     - `--skip-git-repo-check`
     - `--dangerously-bypass-approvals-and-sandbox`

2. **Claims 本地化**
   - 默认不上传 `claims_ir.json` 到沙盒。
   - Phase2 仅上传：`task_spec.json`、`metric_contract.json`、repo、data。
   - `run_manifest.json` / `claim_alignment.json` / `codex_worklog.jsonl` 由 Runner 本地组装并写回 artifacts。

3. **依赖安装 Runner 主导**
   - 先做 capability gate（`python3/pip/ensurepip/numpy`）。
   - 再做 dependency bootstrap（支持后台执行+轮询日志）。
   - legacy 依赖支持兼容映射（生成 `requirements.compat.txt`）。
   - 关键日志：
     - `execution/codex_outputs/dependency_bootstrap.log`
     - `execution/codex_outputs/pip_install.log`

4. **串行单任务 + 20 秒流式日志回传**
   - `task_spec.tasks` 按顺序执行：每次只跑 1 个 task（1 次 `codex exec` 会话）。
   - 默认 task 失败后继续下一个 task（不中断整批）。
   - 运行期间每 20 秒从沙盒增量同步 `codex_exec.log` 到本地：
     - `execution/codex_outputs/codex_exec.stream.log`
   - 保留镜像日志：
     - `execution/codex_outputs/codex_exec.log`
     - `execution/codex_outputs/codex_main.log`
     - `execution/codex_outputs/codex_repair.log`

5. **长任务稳定性**
   - 长命令采用后台 + `pid/rc/log` 轮询，减少流式断开误杀。
   - 断流时优先继续轮询，不立即 teardown。
   - Phase2 全局硬超时：`45` 分钟（`GLOBAL_TIMEOUT_45M`）。

6. **输出契约**
   - `execution/codex_outputs/run_manifest.json`
   - `execution/codex_outputs/codex_worklog.jsonl`
   - `execution/codex_outputs/patches.diff`
   - `execution/codex_outputs/claim_alignment.json`
   - 失败时：`execution/codex_failure.json`

### 最近一次实跑快照（run_id=`codex_e2b_001`）

- 模型/API切换后（`gpt-5.1`）可完成整批执行，`5` 个 task 中 `3` 个成功、`2` 个失败。
- 失败任务：
  - `task_01` (`main_gru_svm.py`)
  - `task_04` (`main_mlp.py`)
- 共性报错：`AttributeError: module 'tensorflow' has no attribute 'placeholder'`
  - 属于典型 TF1 API 在 TF2 运行时不兼容。
- 额外观察：
  - `task_05` 有 `TASK_RESULT_MISSING_FROM_CODEX`，说明曾存在“任务执行后结果未完整落盘”的稳定性问题。
- 建议优先查看：
  - `execution/codex_outputs/task_run_results.json`
  - `execution/codex_outputs/run_manifest.json`
  - `execution/codex_outputs/dependency_solver.json`

### Prompt 新增硬限制（已生效）

- 必须为当前 `task_id` 写入一条 `task_run_results.json` 记录，禁止漏写。
- 退出前必须校验 `task_run_results.json` 为合法 JSON 且包含当前 `task_id`。
- 命中 TF1/TF2 API 不兼容（`tf.placeholder`/`tf.set_random_seed`/`tf.contrib`）时，统一标记：
  - `status=failed_dependency`
  - `reason_codes` 包含 `TF1_API_INCOMPATIBLE_WITH_TF2`
- 执行失败时也必须写结构化产物（至少 `task_run_results.json` + `codex_worklog.jsonl`）。

### TaskSpec（Breaking 变更）

`task_spec.json` 使用任务数组，不再使用 `goal`：

```json
{
  "tasks": [
    {
      "task_id": "task_01",
      "entrypoint": "main.py",
      "command": "python3 main.py --epochs 10",
      "timeout_class": "medium",
      "expected_metrics": ["accuracy"],
      "hyperparams": {"epochs": 10}
    }
  ],
  "constraints": {
    "budget_minutes": 30,
    "max_self_heal_iters": 2,
    "network_policy": "default"
  },
  "selection_notes": ["code-verifiable only"]
}
```

### 关键环境变量

- `P2C_RUNTIME_BACKEND=e2b`
- `E2B_API_KEY`
- `OPENAI_API_KEY`
- `P2C_E2B_TEMPLATE`（默认 `openai-codex`）
- `P2C_CODEX_MODEL`（默认 `gpt-5.1`）
- `P2C_E2B_UPLOAD_TMP_DIR`（可选；指定 E2B 上传临时文件目录，建议 `/workspace`）
- `P2C_WORKSPACE_ROOT`（可选；覆盖沙盒工作目录根）
- `P2C_UPLOAD_CLAIMS_TO_SANDBOX`（默认 `0`）
- `P2C_DEP_BOOTSTRAP_ENABLE`（默认 `1`）
- `P2C_DEP_BOOTSTRAP_APT_ENABLE`（默认 `1`）
- `P2C_DEP_BOOTSTRAP_RUNTIME_SUDO_ENABLE`（默认 `1`）
- `P2C_DEP_COMPAT_MODE`（默认 `1`）
- `P2C_DEP_COMPAT_PROFILE`（默认 `tf1_legacy`）
- `P2C_TASK_SERIAL_MODE`（默认 `1`）
- `P2C_TASK_BATCH_SIZE`（默认 `1`）
- `P2C_TASK_CONTINUE_ON_FAILURE`（默认 `1`）
- `P2C_TASK_RATE_LIMIT_RETRIES`（默认 `1`）
- `P2C_TASK_RATE_LIMIT_BACKOFF_SEC`（默认 `60`）
- `P2C_TASK_RATE_LIMIT_BACKOFF_MULTIPLIER`（默认 `2.0`）
- `P2C_STREAM_SYNC_ENABLE`（默认 `1`）
- `P2C_STREAM_SYNC_INTERVAL_SEC`（默认 `20`）
- `P2C_STREAM_LOCAL_PATH`（默认 `execution/codex_outputs/codex_exec.stream.log`）

### 诊断顺序（推荐）

1. `execution/codex_outputs/codex_exec.stream.log`
2. `execution/codex_outputs/codex_exec.log`
3. `execution/codex_failure.json`

### 运行示例

```bash
python -m p2c.main \
  --phase 2 \
  --paper_md Target/paper/full.md \
  --paper_md_out output/paper.md \
  --repo_dir Target/code \
  --run_id codex_e2b_001 \
  --artifacts_dir artifacts \
  --budget_minutes 30 \
  --max_self_heal_iters 2
```

### 下一步（稳定性增强）

1. 失败原因探索：
   - 将失败聚类为 `环境不兼容`、`依赖冲突`、`结果漏写`、`速率限制` 四类，沉淀标准 `reason_codes`。
2. 跨仓库回归：
   - 至少准备 3 类 repo（TF1 老项目 / 现代 PyTorch / 纯 sklearn），固定同一 Phase2 配置跑 A/B。
3. 稳定性门禁：
   - 以“任务结果完整率（每 task 必有记录）”和“可复现失败原因覆盖率”作为发布前门槛。
4. Skills 化方向（可落地）：
   - `phase2-failure-triage`：日志->分类->建议修复动作。
   - `phase2-cross-repo-regression`：批量执行->汇总->回归对比报告。

---

## English

### Current Architecture

This repo currently uses a 3-phase pipeline:

1. `Phase1`: build executable `task_spec.json` from paper/code (task-oriented, no `goal=[claim_ids]`).
2. `Phase2`: run tasks inside E2B sandbox (gitless), then assemble structured outputs locally in the runner.
3. `Phase3`: perform claims evidence alignment and verification locally.

Key direction:

- SWE-agent execution path is removed.
- `E2B + codex exec` is the only execution backend.
- Codex in sandbox is task execution only; claims judgment is local.

### Phase2 Behavior

- Uploads repo/data + `task_spec.json` (+ optional `metric_contract.json`) to sandbox.
- Does **not** upload claims by default.
- Runs capability gate + dependency bootstrap (with background long-command polling).
- Runs tasks in **serial single-task mode** (one `codex exec` session per task).
- Continues to next task on failure by default.
- Streams sandbox log delta every 20 seconds to:
  - `execution/codex_outputs/codex_exec.stream.log`
- Enforces a hard global timeout of `45 minutes`.
- Builds structured outputs locally:
  - `execution/codex_outputs/run_manifest.json`
  - `execution/codex_outputs/codex_worklog.jsonl`
  - `execution/codex_outputs/patches.diff`
  - `execution/codex_outputs/claim_alignment.json`
- Writes `execution/codex_failure.json` on failure with stage-level diagnostics.

### Latest Run Snapshot (`run_id=codex_e2b_001`)

- After switching model/API (`gpt-5.1`), the batch completed end-to-end.
- Result: `5` tasks total, `3` succeeded, `2` failed.
- Failed tasks: `task_01` (`main_gru_svm.py`), `task_04` (`main_mlp.py`).
- Shared error: `AttributeError: module 'tensorflow' has no attribute 'placeholder'` (TF1 API on TF2 runtime).
- Additional signal: `task_05` had `TASK_RESULT_MISSING_FROM_CODEX`, indicating prior output-completeness instability.

### Prompt Hard Constraints (Current)

- Must persist one run record for the current `task_id` in `task_run_results.json`.
- Must validate JSON shape + current `task_id` presence before exit.
- TF1-vs-TF2 API errors (`tf.placeholder` / `tf.set_random_seed` / `tf.contrib`) must be normalized as:
  - `status=failed_dependency`
  - `reason_code=TF1_API_INCOMPATIBLE_WITH_TF2`
- Even on execution failure, structured outputs must still be written.

Recommended debug order:
1. `execution/codex_outputs/codex_exec.stream.log`
2. `execution/codex_outputs/codex_exec.log`
3. `execution/codex_failure.json`

### Phase2 Codex Flags

Codex runs with fixed flags:

- `--skip-git-repo-check`
- `--dangerously-bypass-approvals-and-sandbox`

Default model:

- `gpt-5.1` (override via `P2C_CODEX_MODEL`).

### Next Steps

1. Build a failure taxonomy (`env incompat`, `dependency conflict`, `missing task result`, `rate limit`).
2. Run cross-repo regression on at least three repo types (legacy TF1, modern PyTorch, pure sklearn).
3. Gate releases with output completeness + failure-reason reproducibility.
4. Potential skills to extract:
   - `phase2-failure-triage`
   - `phase2-cross-repo-regression`

### TaskSpec Breaking Change

`TaskSpec` now uses `tasks[]` (execution-first schema) instead of `goal`.
Only code-verifiable objectives should be turned into tasks/metrics/hyperparameters.
