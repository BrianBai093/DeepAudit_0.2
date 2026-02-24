from __future__ import annotations

from textwrap import dedent


def build_codex_main_prompt(
    max_self_heal_iters: int,
    repo_dir: str,
    inputs_task_spec: str,
    inputs_claims_ir: str,
    outputs_dir: str,
) -> str:
    return dedent(
        f"""
        You are running inside an E2B sandbox.
        Hard constraints:
        1) Work only in {repo_dir}
        2) Read {inputs_task_spec} and {inputs_claims_ir} first
        3) Run dependency_solver before execution (Codex-led but structured):
           - Detect dependency source in this order: pyproject.toml, requirements*.txt, environment.yml, setup.py
           - Decide install plan and command order
           - Record install events to codex_worklog.jsonl with type=install, command, result, and error summary
           - If pip is used, write install logs to {outputs_dir}/pip_install.log
           - Write dependency decisions and outcomes to {outputs_dir}/dependency_solver.json
           - Never silently retry unresolved install conflicts; write reason_codes and move on
        4) Execute repo according to task_spec entrypoints/run_matrix
        5) If one entrypoint cannot run due to dependency conflict, keep going for other entrypoints and record status+reason_codes in run_manifest
        6) If all entrypoints are unrunnable due to dependencies, still write structured outputs and explicit dependency failure reason_codes
        7) If execution fails, do minimal code fixes and retry (max {max_self_heal_iters} retries)
        8) Write ALL required outputs to {outputs_dir}:
           - run_manifest.json
           - codex_worklog.jsonl
           - patches.diff
           - claim_alignment.json
        9) claim_alignment MUST be concept-level only; do NOT include final numeric claim judgments.
        10) The repo may not be a git repository. Do NOT stop because of git state.

        JSON contracts:
        - run_manifest.json: {{"runs": [...], "reason_codes": [...]}}
        - claim_alignment.json: {{"claims": [...], "reason_codes": [...]}}

        For each run_manifest item include:
        run_id, command, params, cwd, exit_code, status, runtime_sec,
        stdout_tail, stderr_tail, artifacts, metrics, reason_codes.

        For each claim_alignment item include:
        claim_id, required_metrics, source, evaluable(yes/no/partial), reason.

        Produce deterministic, machine-readable outputs.
        """
    ).strip()


def build_codex_repair_prompt(outputs_dir: str) -> str:
    return dedent(
        f"""
        Repair-only mode.
        Do NOT run training/benchmark again.
        Preconditions:
        - Use this mode only when execution artifacts already exist in {outputs_dir}.
        - If outputs are entirely missing, do NOT use repair-only; a main execution is required first.
        - Do not rerun full dependency installation or training workloads in this mode.
        Only read existing outputs and logs, then fix/complete:
        - {outputs_dir}/run_manifest.json
        - {outputs_dir}/claim_alignment.json
        - {outputs_dir}/codex_worklog.jsonl
        - {outputs_dir}/patches.diff

        Keep schemas valid and keep claim_alignment concept-only.
        """
    ).strip()
