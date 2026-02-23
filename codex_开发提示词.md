## Codex 开发提示词（优化版，三阶段可测试交付）

你是资深智能体系统工程师 + vibe coding 专家。请在当前仓库实现 **Paper2Code：论文复现与结论验证（Claim Verification）** 的端到端 MVP（分三阶段交付，每阶段代码必须可运行可测试）。系统目标：输入带图片的 markdown 论文与本地已 clone 的 repo，自动抽取论文指标与结论（Claim IR），生成可执行 TaskSpec，调用 mini-SWE 在沙盒/本机环境执行最小可验证运行（MVR），解析指标并验证是否支持论文结论，输出可追溯 artifacts 与报告。

### 全局硬性要求（必须遵守）

1. **三阶段开发，每阶段都能直接测试**：不得提交仅有接口/占位符而无法运行的代码。每阶段完成后必须提供可运行 CLI 与最少测试用例。
2. **每个 Agent 都必须接入 LLM（OpenAI API）**：每个 Agent 调用必须通过统一的 `LLMClient`（或等价）封装；读取 `OPENAI_API_KEY`（以及可选 `OPENAI_BASE_URL`, `OPENAI_MODEL`）。我已配置在环境中
3. **每个 Agent 工作时必须在命令行输出进度与状态**：采用统一日志格式（建议：`[agent=<name>] [state=<...>] [step=i/n] message`），并同时写入 artifacts 的 `execution/run.log`。
4. **向用户提问：遇到不确定时向用户提问**
5. **不改变 artifacts 目录与关键字段命名**：以“端到端开发流程（修订版）”为真源；所有阶段都必须落地完整 artifacts 树（即便部分字段为空也要可解释）。
6. **输入约束**：输入是 `Target/paper/full.md`（包含图片），先用既有资产 `PictureToWords.py` 将图片替换为文字描述，输出为 `output/paper.md`；后续 Agent 一律以 `output/paper.md` 作为论文输入。
7. 在你的skills库中，由用于langgraph开发的skills，合理利用

---

## 0) 代码结构（必须按此创建/落地）

创建包 `p2c/`，并把每个 Agent 作为独立模块 + 统一基类（强制使用 OpenAI API）：

- `p2c/main.py`：CLI 入口（支持 `--phase 1|2|3`）
- `p2c/graph.py`：LangGraph（可选：Phase 1 先用线性 pipeline，但仍要保留未来接 LangGraph 的结构）
- `p2c/llm/client.py`：OpenAI API 封装（所有 Agent 强制使用）
- `p2c/agents/base.py`：Agent 基类（进度输出、结构化结果、重试）
- `p2c/agents/ingest_paper.py`：节点1 ingest\_paper
- `p2c/agents/extract_fingerprint.py`：节点2 extract\_fingerprint
- `p2c/agents/build_claims_ir.py`：节点3 build\_claims\_ir
- `p2c/agents/compile_task_spec.py`：节点4 compile\_task\_spec
- `p2c/agents/prepare_sandbox.py`：节点5 prepare\_sandbox
- `p2c/agents/setup_env.py`：节点6 setup\_env
- `p2c/agents/resolve_data.py`：节点7 resolve\_data（MVP 可做“声明式清单 + 可解释占位”，但必须可运行）
- `p2c/agents/execute_and_heal.py`：节点8 execute\_and\_heal（调用 mini-SWE）
- `p2c/agents/observe_metrics.py`：节点9 observe\_metrics（按 metric\_contract 解析）
- `p2c/agents/align_evidence.py`：节点10 align\_evidence
- `p2c/agents/verify_claims.py`：节点11 verify\_claims
- `p2c/agents/audit_report.py`：节点12 audit\_report（生成 report.md 并调用 PictureToWords）
- `p2c/io_artifacts.py`：artifacts 读写、原子写、run\_id、hash
- `p2c/schemas.py`：Pydantic schema（所有节点 I/O 强校验）
- `p2c/utils/console.py`：统一控制台日志格式
- `tests/`：阶段化测试（每阶段至少 2 个）

说明：

- **Phase 1** 完成节点 1–4（PictureToWords→指标/Claim 抽取→TaskSpec）
- **Phase 2** 完成节点 5–8（编译 TaskSpec→下命令给 mini-SWE→执行并汇总）
- **Phase 3** 完成节点 9–12（指标解析→证据对齐→结论验证→报告）

---

## 1) 统一 CLI（必须实现，分阶段可运行）

实现命令（所有阶段通用）：

```bash
python -m p2c.main \
  --phase 1 \
  --paper_md Target/paper/full.md \
  --paper_md_out output/paper.md \
  --repo_dir Target/code \
  --run_id demo_run_001 \
  --artifacts_dir ./artifacts \
  --budget_minutes 60 \
  --max_self_heal_iters 6
```

约束：

- `--phase 1`：只跑 PictureToWords + 节点1-4
- `--phase 2`：跑节点5-8（要求 phase1 artifacts 已存在；若不存在则报错并提示如何运行 phase1）
- `--phase 3`：跑节点9-12（要求 phase2 metrics/logs 已存在；若不存在则报错并提示如何运行 phase2）
- 每次运行都必须生成：`/artifacts/{run_id}/...` 完整树（文件必须存在，字段可为空但要可解释 reason\_codes）。

---

## 2) 全局 artifacts 规范（每阶段都必须落地）

统一根目录：`/artifacts/{run_id}/` 必须生成（至少为空结构 + reason）：

- `paper/paper_text.json`
- `paper/citations.json`
- `fingerprint/fingerprint.json`
- `fingerprint/claims_ir.json`
- `task/task_spec.json`
- `task/metric_contract.json`
- `execution/run.log`
- `execution/commands.jsonl`
- `execution/patch.diff`
- `execution/repo_state.json`
- `execution/system_info.json`
- `execution/env_lock/`（至少 `pip_freeze.txt`）
- `execution/data_manifest.json`
- `results/metrics.json`
- `results/parsed_evidence.json`
- `results/verdict.json`
- `results/report.md`

并且：

- 所有 Agent 的结构化输出都必须通过 Pydantic 校验；失败要写入 `run.log`，并在 `verdict.json` 标记 `INCONCLUSIVE + reason_codes`（Phase1/2 也要生成 verdict.json，但可标记 `INCONCLUSIVE`）。

---

## 3) OpenAI API 接入规范（每个 Agent 强制）

实现 `LLMClient`：

- 从环境变量读取：
  - `OPENAI_API_KEY`（必需）
  - `OPENAI_MODEL`（默认例如 `gpt-5-mini` 或仓库内统一常量）
  - `OPENAI_BASE_URL`（可选）
- 提供最少两个方法：
  - `chat_json(schema: dict, system: str, user: str) -> dict`：要求模型输出严格 JSON（用于 schema 化抽取）
  - `chat_text(system: str, user: str) -> str`

每个 Agent 的 Prompt 必须：

- 明确输入文件路径、输出文件路径、schema 要求
- 明确 “不得编造；不确定必须写 reason\_codes/notes”
- 明确 “输出必须是 JSON，字段必须完整（可空但要解释）”
- 明确 “在 console 输出进度：开始/关键步骤/结束/耗时”

---

## 4) 三阶段交付要求（具体到节点与验收）

### Phase 1（必须完成并可测试）：PictureToWords → 节点1-4

目标：从 `Target/paper/full.md` 生成 `output/paper.md`，并产出可用于后续执行的 `claims_ir.json` 与 `task_spec.json`。

实现节点：

1. `ingest_paper`
   - 输入：`output/paper.md`
   - 输出：`paper/paper_text.json`（至少含：sections 列表、raw\_text、figure\_descriptions）
   - 同时生成：`paper/citations.json`（至少可为空数组，但要有 schema）
2. `extract_fingerprint`
   - 输出：`fingerprint/fingerprint.json`（paper\_meta、datasets、evaluation\_setup 等最小字段齐全）
3. `build_claims_ir`（关键）
   - 输出：`fingerprint/claims_ir.json`
   - 必须含字段：`claim_id,type,predicate,metric,target,baseline,conditions,aggregation,evidence_set,tolerance_policy`
   - 若无法定位证据：标记 `unverifiable_from_paper=true` 并填原因
4. `compile_task_spec`（关键）
   - 输出：`task/task_spec.json`，至少包含：
     - `goal`（claim\_id 列表）
     - `constraints`（预算/联网/修改范围）
     - `entrypoints[]`（Phase1 可以从 repo\_dir 扫描 README/scripts/pyproject/CI 文件，生成 Top-5 候选，但必须真实存在的文件线索）
     - `metric_observers[]`（至少 stdout\_regex 方案）
     - `run_matrix`（MVR：至少 1 个 seed，预算字段要有）
   - 同时生成：`task/metric_contract.json`（required\_metrics + parsers + normalization）

Phase 1 测试（必须写入 `tests/`）：

- `test_claims_ir_schema_valid()`：对 `claims_ir.json` 做 Pydantic 校验
- `test_task_spec_has_entrypoints_and_observers()`：确保 entrypoints<=5，且含至少一个 observer

**禁止**：Phase1 只生成空文件或全靠 mock。必须真实运行 PictureToWords、真实解析 markdown、真实调用 LLM 生成 claims/task。

---

### Phase 2（必须完成并可测试）：节点5-8 + mini-SWE 执行与汇总

目标：`compile_task_spec` 生成的入口/命令被转换为 mini-SWE 的“高层任务”，并在 `Target/code` 下运行；产出可解析的运行结果与日志。

实现节点： 5. `prepare_sandbox`

- 输出：`execution/system_info.json`（可用 `platform`, `psutil` 等获取）

6. `setup_env`
   - 输出：`execution/env_lock/pip_freeze.txt`（实际执行 `pip freeze` 或 uv/conda 信息）
7. `resolve_data`
   - 输出：`execution/data_manifest.json`（即便无法下载，也要写明 `unresolved` + reason\_codes；不可静默跳过）
8. `execute_and_heal`（关键）
   - 读取：`task/task_spec.json` 的首选 entrypoint + run\_matrix
   - 生成给 mini-SWE 的“任务指令”（包含：工作目录=Target/code；执行命令；成功标准=生成 metrics 线索或日志中出现指标）
   - 必须落地：
     - `execution/commands.jsonl`（结构化记录每条命令、cwd、rc、摘要）
     - `execution/run.log`（拼接 stdout/stderr）
     - `execution/patch.diff`（若 mini-SWE 修改了代码）
     - `execution/repo_state.json`（commit、diff、submodules；本地 repo 也要记录当前 HEAD）
   - 安全约束必须做“逻辑层拒绝”：拦截 sudo、curl|bash 等（即便不是真 sandbox 也要拒绝）
   - 执行结果汇总：在 `results/metrics.json` 写入**至少一条 record**（哪怕是 “unparsed” 也要有 reason\_codes）

Phase 2 测试（必须）：

- `test_commands_jsonl_written()`：跑一个最小 dummy 命令（例如 `python -c "print('ok')"`）确保 commands.jsonl 正确写入
- `test_patch_diff_exists_even_if_empty()`：patch.diff 文件必须存在

---

### Phase 3（必须完成并可测试）：节点9-12 结论验证与报告

目标：从运行日志/metrics 解析出指标 → 映射到 claim → 输出四态 verdict + report。

实现节点： 9. `observe_metrics`

- 按 `task/metric_contract.json` 解析 `execution/run.log` 或 json/csv 输出
- 更新 `results/metrics.json`（records 标准化）

10. `align_evidence`

- 输出：`results/parsed_evidence.json`（每条 claim 对齐到哪些 records，缺失原因是什么）

11. `verify_claims`（关键：四态）

- 输出：`results/verdict.json`
- `status ∈ {SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE}`
- 实现最小判定：
  - absolute：`|x_rep-x_paper|<=max(abs_eps, rel_eps*|x_paper|)`
  - relative/improves\_by：`delta_rep >= delta_paper - eps`
  - ranking/argmax：Top-1/Top-k 一致性（允许 tie 需说明）
- 统计不足/数据不可得/预算不足 → **必须 INCONCLUSIVE**（不得当 NOT\_SUPPORTED）

12. `audit_report`

- 输出：`results/report.md`（包含：repo commit、环境、数据清单、命令轨迹、指标汇总、逐 claim 判定、不确定性来源）
- 生成 report 前必须运行 `PictureToWords.py`：若 report/log 引用了 markdown 图片，附加“图片文字化描述”附录

Phase 3 测试（必须）：

- `test_absolute_claim_verdict()`：构造一条 absolute claim + metrics，验证 SUPPORTED/NOT\_SUPPORTED 分支
- `test_inconclusive_when_missing_records()`：缺 records 必须 INCONCLUSIVE

---

## 5) mini-SWE 集成要求（必须真实可跑）

- 假设 `mini_swe/` 已存在于仓库根目录
- `execute_and_heal` 必须通过子进程调用 mini-SWE（或其可执行入口），并捕获：
  - 运行命令与返回码
  - mini-SWE 的 stdout/stderr 追加进 `execution/run.log`
  - 若产生修改：把 git diff 写入 `execution/patch.diff`
- 若 mini-SWE 不可用/失败：必须降级为“只执行 entrypoint 不修复”，但要在 verdict/report 标记 `INCONCLUSIVE` 并写清原因（不能静默）。

---

## 6) Agent Prompt 设计（必须写在代码里，可调优）

为每个 Agent 在 `p2c/agents/<agent>.py` 内定义：

- `SYSTEM_PROMPT`（固定）
- `USER_PROMPT_TEMPLATE`（带输入输出路径）
- 强制输出格式（JSON schema 或严格字段清单）

示例要求（适用于 build\_claims\_ir / compile\_task\_spec）：

- 输出必须是单个 JSON 对象，不要 markdown
- 字段缺失必须补齐为 null/[] 并添加 `reason_codes`
- 不得编造数值：论文里找不到就标记 `unverifiable_from_paper`

---

## 7) 日志与可观测性（必须）

- 控制台：每个 Agent 必须打印：
  - START（输入/输出路径）
  - PROGRESS（关键子步骤）
  - DONE（耗时、产物路径）
- artifacts：
  - `execution/run.log`：包含所有 Agent 的关键日志 + 命令输出
  - `execution/commands.jsonl`：结构化命令轨迹（时间、cwd、cmd、rc、stdout/stderr 摘要、资源字段可 null）

---

## 8) 交付与质量门槛（必须做到）

- 任何阶段失败也要输出完整 artifacts 树
- 所有 schema 校验失败必须写入 run.log 并体现在 verdict.json（INCONCLUSIVE + reason）
- 不硬编码特定论文/repo；一切从输入路径与 repo\_dir 扫描得到
- 每阶段结束后，确保 `python -m p2c.main --phase <n> ...` 可直接运行



