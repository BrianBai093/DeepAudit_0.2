"""Prompt templates for local Codex execution in Phase 2."""

from __future__ import annotations

from textwrap import dedent
from typing import Any


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = dedent("""\
    You are an expert code-execution planner for ML/DL research repositories.
    Given a paper's claimed results, the repository structure, and dependency manifests,
    produce a precise ExecutionPlan JSON that another agent will follow to reproduce
    the paper's results on a local machine.
    Return ONLY valid JSON matching the schema below — no markdown fences, no commentary.
""").strip()


def build_planner_user_prompt(
    *,
    claims_ir_json: str,
    task_spec_json: str,
    metric_contract_json: str,
    repo_analysis_json: str,
    repo_tree: str,
    readme_content: str,
    dependency_files: dict[str, str],
    failure_context: str | None = None,
    env_name: str,
    budget_sec: int,
) -> str:
    dep_section = "\n".join(
        f"### {name}\n```\n{content}\n```" for name, content in dependency_files.items()
    )
    failure_section = ""
    if failure_context:
        failure_section = f"""
## Previous Execution Failures (AVOID repeating these mistakes)
```json
{failure_context}
```
"""

    return dedent(f"""\
## Paper Claims to Reproduce
```json
{claims_ir_json}
```

## Repository Analysis
```json
{repo_analysis_json}
```

## Task Specification
```json
{task_spec_json}
```

## Metric Extraction Contract
```json
{metric_contract_json}
```

## Repository File Tree (first 500 entries)
```
{repo_tree}
```

## Key File Contents
### README.md
```
{readme_content}
```

{dep_section}
{failure_section}

## Output Requirements
Return a JSON object with this exact schema:
{{
  "plan_id": "<unique string>",
  "plan_version": <int, start at 1, increment on replan>,
  "python_version": "<e.g. 3.10>",
  "conda_dependencies": [
    {{"package": "<name>", "version_constraint": "<or null>", "channel": "<defaults|conda-forge|pytorch>", "pip_fallback": <bool>}}
  ],
  "pip_dependencies": ["<raw pip specifiers>"],
  "system_packages": ["<apt package names if needed>"],
  "pre_install_commands": ["<shell commands to run before dependency install>"],
  "execution_steps": [
    {{
      "step_id": "<unique>",
      "description": "<what this step does>",
      "command": "<shell command to run>",
      "cwd": "<relative to repo root, default '.'>",
      "timeout_sec": <int, default 600>,
      "depends_on": ["<step_ids>"],
      "expected_metrics": ["<metric names this step should produce>"],
      "is_setup": <bool, true for data download/preprocessing>,
      "retry_on_failure": <bool>,
      "fallback_commands": ["<alternative commands if main fails>"]
    }}
  ],
  "expected_results": [
    {{
      "claim_id": "<from claims_ir>",
      "metric_name": "<metric to capture>",
      "target_value": <float or null>,
      "extraction_hint": "<how to find it in stdout/files>"
    }}
  ],
  "compatibility_issues": [
    {{"issue_type": "<python_version|cuda_version|package_conflict|os_dependency|other>", "description": "<desc>", "resolution": "<fix>"}}
  ],
  "env_name": "{env_name}",
  "codex_autonomous_fallback": true,
  "total_budget_sec": {budget_sec},
  "reason_codes": [],
  "notes": "<any additional notes>"
}}

Guidelines:
1. Order execution_steps: data download/setup first, then training/evaluation.
2. Set is_setup=true for data download, preprocessing, tokenization steps.
3. Derive python_version from the repo's setup.py/pyproject.toml/CI config; default to 3.10.
4. Include ALL transitive dependencies; prefer pip_dependencies for PyPI packages.
5. For PyTorch/CUDA repos, use the pytorch conda channel with appropriate CUDA version.
6. Each execution step's command must be a single shell command runnable in bash.
7. Set realistic timeout_sec (data download: 300-900s, training: 600-3600s, eval: 120-600s).
8. Map every code-verifiable claim from claims_ir to an expected_result entry.
9. IMPORTANT: In commands, always use `python` (not `python3`) — `python3` may resolve to the system interpreter rather than the conda/venv environment's Python.
10. When tensorflow/torch is a pip dependency, put numpy in pip_dependencies too (not conda) to avoid C ABI mismatches.
""").strip()


# ---------------------------------------------------------------------------
# Codex executor prompts (Mode A — plan-directed, per-step)
# ---------------------------------------------------------------------------

def build_step_execution_prompt(
    *,
    repo_dir: str,
    step_description: str,
    step_command: str,
    expected_metrics: list[str],
    metric_parsers: list[dict[str, Any]],
    outputs_dir: str,
    step_id: str,
    prior_step_results: str | None = None,
) -> str:
    parsers_desc = "\n".join(
        f"  - {p.get('metric_name', '?')}: regex `{p.get('regex', '')}`"
        for p in metric_parsers
    ) or "  (none specified — use METRIC format below)"

    # Inter-step context: let this step see what prior steps produced
    context_section = ""
    if prior_step_results:
        context_section = f"""
## Prior Step Results (READ THIS FIRST)
The following steps have already been executed. Use their outcomes to inform
your approach — e.g. data paths discovered, packages already installed, errors
already encountered, files already created.
```json
{prior_step_results}
```
"""

    return dedent(f"""\
You are executing code in a research repository to reproduce results from a paper.
Working directory: {repo_dir}
The conda/venv environment is already activated.
{context_section}
## Current Task
{step_description}

## Command to Execute
```bash
{step_command}
```

## Expected Metrics
{', '.join(expected_metrics) if expected_metrics else '(discover any numeric metrics)'}

## Metric Regex Patterns (from paper analysis)
{parsers_desc}

## Output Format
After execution, print each metric on its own line:
METRIC:<metric_name>=<numeric_value>

IMPORTANT: Distinguish train vs validation/test metrics with prefixes:
METRIC:val_accuracy=0.9534
METRIC:train_accuracy=0.9972
METRIC:val_loss=0.1823
METRIC:test_accuracy=0.9685
Use the unprefixed name (e.g. METRIC:accuracy=0.9534) ONLY for the most meaningful
result — typically the validation or test metric, NOT the training metric.

## Rules
1. Run the command above. If it fails, diagnose the error and fix it (max 3 attempts).
2. If a module is missing, install it with `pip install <package>` and retry.
3. If data files are missing, check the README for download instructions and execute them.
4. Do NOT create a virtual environment or conda environment — one is already active.
5. Record all commands you run.
6. After extracting metrics, write the complete result to:
   {outputs_dir}/step_{step_id}_result.json
   Schema: {{"command": "<final command>", "exit_code": <int>, "metrics": {{"name": value}}, "notes": "<any notes>"}}
7. IMPORTANT: Always use `python` (not `python3`) to run scripts — `python3` may resolve to the system interpreter outside the active conda/venv environment.
""").strip()


# ---------------------------------------------------------------------------
# Codex executor prompts (Mode B — autonomous exploration fallback)
# ---------------------------------------------------------------------------

def build_autonomous_exploration_prompt(
    *,
    repo_dir: str,
    failure_history_json: str,
    expected_results_json: str,
    outputs_dir: str,
    env_path: str | None = None,
) -> str:
    return dedent(f"""\
You are in a research repository and need to reproduce results claimed in a paper.
Previous execution plans have failed. You now have full autonomy to explore and run the code.

## Repository Directory
{repo_dir}
The conda/venv environment is already activated.

## Previous Failure History
```json
{failure_history_json}
```

## Metrics to Extract
```json
{expected_results_json}
```

## Your Task
1. Explore the repository structure (ls, cat README, etc.).
2. Read the README and any documentation for setup/run instructions.
3. Install any missing dependencies with `pip install`.
4. Find and run the code that produces the target metrics.
5. Extract all numeric results you can find.

## Output Format
For each metric found, print:
METRIC:<metric_name>=<numeric_value>

IMPORTANT: Distinguish train vs validation/test metrics with prefixes:
METRIC:val_accuracy=0.9534
METRIC:train_accuracy=0.9972
Use the unprefixed name (e.g. METRIC:accuracy=0.9534) ONLY for the most meaningful
result — typically the validation or test metric, NOT the training metric.

After you are done, write all results to:
{outputs_dir}/autonomous_results.json
Schema: {{
  "commands_run": ["<list of commands you executed>"],
  "metrics": {{"metric_name": value}},
  "notes": "<what you discovered>"
}}

## Rules
1. Maximum 5 execution attempts total.
2. Do NOT create a virtual environment.
3. If you cannot reproduce a metric, note why.
4. Keep output compact — no large file dumps.
5. Always use `python` (not `python3`) to run scripts — `python3` may resolve to the system interpreter outside the active environment.
""").strip()


# ---------------------------------------------------------------------------
# Codex recovery prompt (Mode A2 — direct execution failed, Codex diagnoses)
# ---------------------------------------------------------------------------

def build_codex_recovery_prompt(
    *,
    repo_dir: str,
    step_description: str,
    step_command: str,
    expected_metrics: list[str],
    metric_parsers: list[dict[str, Any]],
    outputs_dir: str,
    step_id: str,
    direct_stdout: str,
    direct_stderr: str,
    direct_exit_code: int,
    env_path: str | None = None,
    prior_step_results: str | None = None,
) -> str:
    parsers_desc = "\n".join(
        f"  - {p.get('metric_name', '?')}: regex `{p.get('regex', '')}`"
        for p in metric_parsers
    ) or "  (none specified — use METRIC format below)"

    context_section = ""
    if prior_step_results:
        context_section = f"""
## Prior Step Results
```json
{prior_step_results}
```
"""

    env_section = ""
    if env_path:
        env_section = f"""
## Environment
The conda environment is at: {env_path}
Use `{env_path}/bin/python` if `python` is not on PATH.
"""

    return dedent(f"""\
You are diagnosing and fixing a failed command in a research repository.
Working directory: {repo_dir}
{env_section}{context_section}
## Failed Command
```bash
{step_command}
```
Exit code: {direct_exit_code}

## Task Description
{step_description}

## stdout (last 3000 chars)
```
{direct_stdout}
```

## stderr (last 3000 chars)
```
{direct_stderr}
```

## Expected Metrics
{', '.join(expected_metrics) if expected_metrics else '(discover any numeric metrics)'}

## Metric Regex Patterns
{parsers_desc}

## Your Task
1. Diagnose why the command failed from the stdout/stderr above.
2. Fix the issue (install missing package, fix path, adjust config, etc.).
3. Re-run the command or an equivalent that produces the expected metrics.
4. Extract metrics and print them as: METRIC:<metric_name>=<numeric_value>
5. Write results to: {outputs_dir}/step_{step_id}_result.json
   Schema: {{"command": "<final command>", "exit_code": <int>, "metrics": {{"name": value}}, "notes": "<what you fixed>"}}

## Rules
1. Maximum 3 fix attempts.
2. Do NOT create a virtual environment — one is already active.
3. Always use `python` (not `python3`).
""").strip()
