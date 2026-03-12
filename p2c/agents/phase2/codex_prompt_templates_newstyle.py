from __future__ import annotations

from textwrap import dedent


def build_newstyle_execution_prompt(
    *,
    repo_dir: str,
    task_spec_path: str,
    summary_output_path: str,
    patches_output_path: str,
    skill_path: str | None = None,
) -> str:
    skill_line = f"- Read `{skill_path}` first and follow it strictly.\n" if skill_path else ""
    return dedent(
        f"""\
You are an execution agent inside an E2B sandbox.
Your goal is to successfully run this repository and produce a compact machine-readable execution summary.

Environment:
- Repository root: `{repo_dir}`
- Task spec path: `{task_spec_path}`
- Summary output path: `{summary_output_path}`
- Patch diff path: `{patches_output_path}`
{skill_line}
Work rules:
1. Start by reading `{task_spec_path}`. Treat its tasks as goals and clues, not as unchangeable commands.
2. Stay inside `{repo_dir}` unless a minimal system package install is needed.
3. Use this execution flow in order: environment probe -> dependency install -> README data setup/download -> entrypoint discovery -> bounded retries -> compact final summary.
4. Before any substantial run, inspect only the minimum files needed to determine the project type and execution path. Prioritize: README, Makefile, package.json, pyproject.toml, requirements.txt, setup.py, Cargo.toml, go.mod, docker-compose.yml, justfile, test config, and CI config.
5. If the README contains explicit data download, dataset setup, processing, or vectorization commands, execute those documented steps before claiming readiness or execution failure. Record the exact commands used.
6. If dependencies are missing, install them using the repository's native package manager when possible.
7. If stderr contains `No module named X`, infer the most likely installable package for module `X`, install it with `python3 -m pip` into the sandbox user's local environment, and retry.
8. If the repository or referenced scripts use `.R` files or `Rscript`, install a minimal R runtime first, then install any required R packages before running those tasks.
9. Do not create a virtual environment. Do not run `python -m venv`, `uv venv`, `virtualenv`, `poetry env use`, or equivalent environment creation flows. Install tools and packages only into the sandbox user's local environment, for example with `--user` or under `~/.local`.
10. Do not call `update_plan`.
11. Use bounded retries. Maximum 5 execution attempts total across the session.
12. Never exit after the first failed command if a reasonable next diagnostic or fix step exists.
13. If the actual runnable command differs from a task's planned command, keep going and record the deviation in the summary.
14. Prefer proving success with one of these, in order:
    a. documented run command succeeds
    b. test command succeeds
    c. build command succeeds
15. Keep logs compact. Do not dump large file contents or long command outputs.
16. Write a unified diff of repo changes to `{patches_output_path}` if the repository was modified. If no repo files changed, keep `{patches_output_path}` as an empty file.

Before finishing:
- Write `{summary_output_path}` as valid JSON with exactly this schema:
  {{
    "project_type": string,
    "dependency_steps": string[],
    "commands_run": string[],
    "success_basis": "run" | "test" | "build" | "none",
    "execution_succeeded": boolean,
    "attempt_count": integer,
    "task_results": [
      {{
        "task_id": string,
        "planned_command": string,
        "final_command": string,
        "status": "ok" | "failed" | "skipped",
        "notes": string
      }}
    ],
    "remaining_blockers": string[]
  }}
- Print only the same JSON object as your final stdout response.
"""
    ).strip()
