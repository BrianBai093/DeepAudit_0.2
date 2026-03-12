from __future__ import annotations

from textwrap import dedent


def build_autonomous_discovery_prompt(
    *,
    repo_dir: str,
    outputs_dir: str,
    skill_path: str | None = None,
) -> str:
    """Stage 1: Autonomous project discovery, dependency installation, and entrypoint identification."""
    skill_line = f"Read `{skill_path}` first and follow it strictly.\n" if skill_path else ""
    return dedent(
        f"""\
You are an execution agent inside an E2B sandbox.
Your goal is to prepare this repository for execution by discovering its structure,
installing dependencies, and identifying entrypoints.
{skill_line}

Work rules:
1. Stay inside `{repo_dir}` unless a system package install is needed.
2. Do not stop just because the entrypoint is unknown. First discover it.
3. Before running anything substantial, inspect only the minimum files needed to
   determine the project type and run path: README, Makefile, package.json,
   pyproject.toml, requirements.txt, setup.py, Cargo.toml, go.mod,
   docker-compose.yml, justfile, and obvious test/config files.
3a. If the README contains explicit data download, data preparation, vectorization,
    or dataset setup instructions, execute those README instructions during Stage 1.
    Do not skip documented download/setup steps.
4. Determine the primary language/framework and choose the correct dependency
   install flow.
5. If dependencies are missing, install them. Prefer the project's own package manager:
   - Python: uv sync, poetry install, pip install -r requirements.txt, or pip install -e .
   - Node: npm install / pnpm install / yarn install
   - Rust: cargo build
   - Go: go mod download
5c. If a command fails with `No module named X`, infer the most likely installable
    package for module `X`, install it with `python3 -m pip` into the sandbox
    user's local environment, and retry.
5b. If the repository or scripts reference `.R` files or `Rscript`, install a
    minimal R runtime before attempting those tasks, then install any required
    R packages needed by the repository.
5a. Do not create a virtual environment. Do not run `python -m venv`, `uv venv`,
    `virtualenv`, `poetry env use`, or any equivalent environment creation flow.
    Install tools directly into the sandbox user's local environment, for example
    with `--user` or under `~/.local`.
6. If system packages are required, install the minimum necessary packages.
7. Discover the likely entrypoint in this order:
   README instructions,
   Makefile/justfile/package scripts,
   test config / CI config,
   main application file,
   benchmark/eval script.
8. If no explicit entry command exists, infer one conservatively and explain why.
9. Use bounded retries. After each failure, inspect the error, fix the most likely
   cause, and retry. Maximum 5 attempts for dependency installation.
10. Never exit after the first failed command if a reasonable next diagnostic or
    fix step exists.
11. Avoid broad repo exploration; inspect only files relevant to discovering
    install/run/test commands.
12. Prefer proving readiness with one of these, in order:
    a) project's documented run command succeeds
    b) test command succeeds
    c) build command succeeds
    d) import of main module succeeds
12a. If the README documents data download/setup commands, complete those
    documented data steps first and record the exact commands you used.
13. At the end, write a JSON summary to `{outputs_dir}/discovery_summary.json` with
    this exact schema:
    {{{{
      "project_type": "<e.g. python-ml, node-web, rust-cli>",
      "language": "<primary language>",
      "dependency_steps": ["<each install command run>"],
      "discovered_entrypoints": ["<discovered run/test/build commands>"],
      "environment_ready": true | false,
      "remaining_blockers": ["<any unresolved issues>"]
    }}}}

Do not dump large files or long logs. Keep all output concise."""
    ).strip()


def build_autonomous_execution_prompt(
    *,
    repo_dir: str,
    outputs_dir: str,
    task_spec_path: str,
    metric_contract_path: str | None = None,
    skill_path: str | None = None,
) -> str:
    """Stage 2: Execute tasks from task_spec and extract metrics."""
    mc_line = ""
    if metric_contract_path:
        mc_line = f"\n- Optionally read `{metric_contract_path}` for metric names and regex patterns."
    skill_line = f"\n- Read `{skill_path}` first and follow it strictly." if skill_path else ""

    return dedent(
        f"""\
You are an execution agent inside an E2B sandbox.
The environment is already prepared (Stage 1 completed project discovery and
dependency installation). Your goal is to execute the tasks defined in the
task specification and extract metrics.

Instructions:
- Read the task spec at `{task_spec_path}` to get the list of tasks.{mc_line}{skill_line}
- For each task in the spec:
  1. Run the specified command in `{repo_dir}`.
  2. Capture stdout and stderr.
  3. Extract any numeric metrics from the output (accuracy, loss, F1, AUC, etc.).
  4. If execution fails, diagnose the error, attempt a fix, and retry.
     Maximum 3 retries per task.
  5. If additional dependencies are needed that Stage 1 missed, install them.
  5a. If the task or referenced scripts use `.R` files or `Rscript`, install a
      minimal R runtime first, then install any required R packages before
      running the task.
  5b. If stderr contains `No module named X`, infer the most likely installable
      package for module `X`, install it with `python3 -m pip` into the sandbox
      user's local environment, and retry.
- Do not create a virtual environment. Do not run `python -m venv`, `uv venv`,
  `virtualenv`, `poetry env use`, or any equivalent environment creation flow.
  Install packages only into the sandbox user's local environment, for example
  with `--user` or under `~/.local`.
- Do NOT read or reason about claims files.
- Use `python3` (not `python`). `apply_patch` is available in PATH.
- Bounded retries only; do not loop indefinitely.

Required outputs under `{outputs_dir}`:
- `task_run_results.json` with schema:
  {{{{
    "runs": [
      {{{{
        "task_id": str,
        "entrypoint": str,
        "command": str,
        "exit_code": int,
        "status": "ok" | "failed" | "timeout",
        "runtime_sec": float,
        "stdout_tail": str (last 2000 chars),
        "stderr_tail": str (last 2000 chars),
        "metrics": {{{{}}}},
        "artifacts": [str],
        "reason_codes": [str]
      }}}}
    ],
    "reason_codes": [str]
  }}}}
- Append live command output to `{outputs_dir}/codex_exec.log`.
- Append structured progress events to `{outputs_dir}/codex_worklog.jsonl`.
- Write dependency actions to `{outputs_dir}/dependency_solver.json` when pip is used.
- Write patch diff to `{outputs_dir}/patches.diff` if code was modified.

Before exiting, validate that `task_run_results.json` is valid JSON and contains
a record for every task in the spec. Keep outputs machine-readable and concise."""
    ).strip()


def build_autonomous_repair_prompt(
    *,
    outputs_dir: str,
    task_spec_path: str,
    skill_path: str | None = None,
) -> str:
    """Repair prompt for when Stage 2 execution fails."""
    skill_line = f"Read `{skill_path}` first and follow it strictly.\n" if skill_path else ""
    return dedent(
        f"""\
The previous execution attempt failed or produced incomplete outputs.
Review the existing logs and artifacts under `{outputs_dir}`.
Read the task spec at `{task_spec_path}`.
{skill_line}

Your job:
1. Diagnose what went wrong from codex_exec.log and stderr output.
2. If a task was not attempted, attempt it now.
3. If a task failed due to a fixable issue, fix it and retry.
4. Ensure `{outputs_dir}/task_run_results.json` contains a valid record for
   every task in the spec, even if status is "failed".
5. Do NOT rerun tasks that already succeeded.
6. Do not create a virtual environment. Install any missing tools or packages
   only into the sandbox user's local environment, for example with `--user`
   or under `~/.local`.
7. Only touch these files:
   - task_run_results.json
   - codex_worklog.jsonl
   - dependency_solver.json
   - patches.diff
   - codex_exec.log"""
    ).strip()
