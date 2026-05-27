# Phase 2 Repo Execution Notes

Phase 2 的职责是把 Phase 1 识别出的论文实验，放到目标 repo 的隔离运行环境里执行，并产出 Phase 3 可消费的标准证据包。当前代码路径不是旧的 plan/replan 模式，而是 `Phase2Orchestrator -> ToolAgent -> ExecutorAgent`。

## 入口和前置产物

CLI 入口是 `python -m p2c.main --phase 2 ...`。`p2c/main.py` 会先检查 `fingerprint/claims_ir.json` 是否存在、可解析，并且包含非空 `experiments`。Phase 2 由 `p2c/graph.py` 调用 `phase2_orchestrator.run(ctx)`。

Phase 2 主要读取这些 Phase 1 产物：

- `fingerprint/claims_ir.json`: authoritative experiment list; executor 必须为每个 experiment 产出一条结果。
- `task/repo_analysis.json`: dependency profiles、entrypoint candidates、manifest paths。
- `task/metric_contract.json`: stdout 指标解析规则和 required metrics。

## 环境如何安装

环境规格由 `ToolAgent.build_env_spec()` 生成，写到 `execution/executor_env_spec.json`。环境名固定为 `<run_id>_executor`。

安装策略分两条路：

1. 优先使用原生 conda/mamba 环境文件。
   `ToolAgent` 会先看 `repo_analysis.dependency_profiles` 里的 `manager == "conda"`，再扫描常见文件名，例如 `p2c_env.yml`、`environment.yml`、`conda_env.yml`、`env.yaml`、`mamba.yaml` 等。找到后执行 `conda/mamba env create -n <env_name> -f <file>`，并覆盖环境文件内的 `name:`。原生环境创建成功后，会跳过从 requirements 推导出来的依赖安装。

2. 没有原生环境文件时，构造派生环境。
   `ToolAgent` 会解析 repo analysis 中的 dependency profiles：
   - `pip_requirements`: 读取 `requirements*.txt`，收集 pip 依赖。
   - `pip_editable`: 记录 `python -m pip install -e .` 作为 post dependency install。
   - `poetry`: 记录 `python -m pip install poetry && poetry install`。
   - conda profile: 解析 yml 里的依赖为 conda dependency。
   - pixi `pyproject.toml`: 解析 `[tool.pixi]` 的 python 版本和 conda-forge 依赖。

实际环境生命周期在 `CondaEnvManager`：

- 优先查找 `mamba`，其次 `conda`。
- 找不到 conda/mamba 时使用 `/tmp/p2c_venv_<env_name>` 的 Python venv fallback。
- conda 路径执行 `conda create -n <env_name> python=<version>`，失败且版本不是 3.10 时会重试 Python 3.10。
- 默认启用 layered install，即按 `core -> ml_libs -> paper_specific` 分层安装；每层安装前做 snapshot，失败则 rollback。核心层失败会中止后续层。
- 每层内部先装 conda 包，再装 pip 包；conda 安装使用 `--freeze-installed` 保护已装核心包，失败后再尝试普通安装和单包 fallback。
- 安装结束后会运行关键 import 校验、`pip freeze`，写入 `execution/env_setup_result.json` 和 `execution/env_lock/pip_freeze.txt`。

如果原生 conda 环境创建失败，`Phase2Orchestrator` 会进入 native env repair 分支：`EnvRepairAgent -> CodeCompatAgent -> ExecutorAgent`。也可以在 phase 2 CLI 中传入 `--phase2_force_env_repair`，跳过原生 yml 的首次直接安装，直接从修复模式开始。普通派生依赖安装失败不进入这个分支，仍会记录 `SOME_PACKAGES_FAILED` / `VALIDATION_FAILED`。

## Repo 如何执行

环境准备通过后，`Phase2Orchestrator` 把 `CondaEnvManager` 放入 `ctx["_p2_env_mgr"]`，然后启动 `ExecutorAgent`。如果 native env repair 被触发，`EnvRepairAgent` 会先把 native yml 解析成 bounded repair candidates，例如放宽 build strings、固定 Python minor、分层安装、CPU PyTorch fallback、conda-forge fallback；`CodeCompatAgent` 再在修复后的环境里做 import-only validation，必要时请求 LLM 生成最小兼容 patch 并直接修改目标 repo。

`ExecutorAgent` 不直接运行固定命令，而是启动一次 Claude Code Agent SDK session，让 executor 根据 repo README、dependency files、repo analysis 和 experiment JSON 自主选择命令。prompt 中强制约束：

- experiment JSON 是唯一实验目标来源。
- repo、README、依赖文件是唯一运行方式来源。
- Python 命令必须使用 managed runtime，例如 `conda run --no-capture-output -n <env> python` 或 venv 下的绝对 `python`。
- pip 命令必须使用 managed pip。
- 不允许修改 repo tracked source/config/script/notebook。
- 审计日志和结果必须写入外部 executor output directory。

标准模式的执行优先级是 `artifact -> smoke -> trend -> full`。如果设置 `P2C_FORCE_FULL_RUN` 或 `P2C_EXECUTION_MODE=full` / `P2C_PHASE2_EXECUTION_MODE=full`，则切换到 full-run 模式，要求优先 fresh full run，不允许先做 smoke/trend 降级。

执行目录处理有一个保护逻辑：如果 canonical artifact 目录位于目标 repo 内，executor 实际看到的是 `/tmp/p2c_executor_outputs/<run_id>/execution/executor_outputs` 或 `P2C_EXECUTOR_OUTPUTS_DIR` 指定目录；session 结束后再同步回 canonical `execution/executor_outputs/`。这样避免 executor 把审计产物写进目标 repo。

## 执行审计和产物

executor session 的宿主侧会持续写：

- `execution/executor_outputs/executor_agent.log`
- `execution/executor_outputs/session_stdout.log`
- `execution/executor_outputs/session_stderr.log`
- `execution/executor_outputs/executor_activity.jsonl`
- `execution/executor_outputs/executor_runtime.json`
- `execution/env_repair/env_repair_result.json`（仅 native env 修复分支）
- `execution/code_compat/code_compat_result.json`（仅 native env 修复分支）
- `execution/code_compat/code_compat_patch.diff`（仅发生代码兼容修改）

executor 被要求为每个 experiment 写：

- `executor_results.json`
- `experiment_<experiment_id>_stdout.log`
- `experiment_<experiment_id>_stderr.log`
- `experiment_<experiment_id>_narrative.log`

session 结束后，`ExecutorAgent` 会：

1. 检查目标 repo tracked files 是否被修改。普通 source/config 修改会让本次执行失败；checkpoint、`.pt`、`.pth`、`.npy` 等 runtime artifact mutation 只记 warning。
2. 选择最新且包含 runs 的 `executor_results*.json`。
3. 对每个 Phase 1 experiment 归一化一条 run：补齐缺失日志、解析 stdout 指标、检查 command 是否真的被 Claude Bash tool 观察到、归一化 `status` / `fidelity` / `evidence_source` / `stop_reason`。
4. 生成 Phase 3 消费的 `execution/executor_outputs/run_manifest.json`。
5. 构建 canonical `execution/executor_outputs/phase2_execution_package.json` 和人类可读 `PHASE2_RESULTS.md`。

Phase 3 优先读取 `phase2_execution_package.json`，`run_manifest.json` 和 raw executor summaries 主要作为同源调试和审计证据。

## 失败与重试

`Phase2Orchestrator` 的最大环境 patch 次数来自 `P2C_MAX_ENV_PATCH`，默认 2。一次 attempt 包含 env setup 和 executor run。executor 失败后，orchestrator 会根据 failure taxonomy 做有限修复：

- `DEP_MISSING_PACKAGE`: 提取缺失包名并 `pip install`。
- `DEP_VERSION_CONFLICT`: 尝试安装基础包名。
- `DEP_CUDA_MISMATCH` / `CFG_WRONG_DEVICE`: 设置 `CUDA_VISIBLE_DEVICES=""`。
- `DEP_BUILD_FAILURE`: 尝试 conda-forge 安装，并允许 pip fallback。

native env repair 分支失败时会写 `execution/env_repair/env_repair_result.json` 和 `execution/execution_failures.json`，并阻止进入 `ExecutorAgent`。如果 `CodeCompatAgent` 的 import-only validation 在 patch 后仍失败，会写 `CODE_COMPAT_FAILED` 并同样阻止进入 `ExecutorAgent`。

Phase 2 结束时如果没有成功 manifest，会写 `execution/execution_failures.json`，并写一个带 `PHASE2_FAILED` reason code 的空 `run_manifest.json`。

## 当前实现边界

- Docker、Node、Make dependency profiles 会被 Phase 1 识别，但 Phase 2 的环境构建主要围绕 conda/venv + Python 依赖；Node/Make 运行更多依赖 executor 自主读取 repo instructions。
- `ExecutionPlan` schema 还保留在 `schemas.py`，但当前 Phase 2 不再以它为执行源。
- README 中提到的 `execution/phase2_artifacts/artifact_storage_preflight.json` / `manifest.json` 路径在当前 `p2c/agents/phase2` 代码中没有落地实现；实际标准证据源是 `execution/executor_outputs/phase2_execution_package.json`。
