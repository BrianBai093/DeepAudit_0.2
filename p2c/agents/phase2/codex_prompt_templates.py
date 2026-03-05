from __future__ import annotations

from textwrap import dedent


def build_codex_single_task_prompt(
    *,
    repo_dir: str,
    inputs_task_spec: str,
    outputs_dir: str,
    task_id: str,
    inputs_metric_contract: str | None = None,
) -> str:
    metric_contract_path = inputs_metric_contract or f"{repo_dir.rstrip('/')}/../inputs/metric_contract.json"
    return dedent(
        f"""
        You are an execution worker running inside an E2B sandbox.
        This is a single-task execution session.
        Hard constraints:
        1) Work only in `{repo_dir}`.
        2) Read only this task spec first: `{inputs_task_spec}`.
        3) Optionally read `{metric_contract_path}` for metric names.
        4) Do NOT read claims files and do NOT perform claim-level reasoning.
        5) Do NOT print or dump full contents of JSON files. Only output compact summaries.
        6) Do NOT perform broad repository exploration. Only open files needed by the current task command.
        7) Execute only task_id=`{task_id}` from the provided task spec.
        8) Bounded retries only; avoid unbounded loops.
        9) Use `python3` only. Do NOT invoke `python`.
        10) Do NOT call `update_plan`.
        11) Patch tool is preinstalled as `apply_patch` in PATH. Prefer `apply_patch` or `python3` file edits.
        12) You MUST persist one run record for task_id=`{task_id}` in `task_run_results.json`. Do not omit task results.
        13) Before exit, validate `task_run_results.json` is valid JSON and contains task_id=`{task_id}`.
        14) If you hit TF1-vs-TF2 API errors (`tf.placeholder`, `tf.set_random_seed`, `tf.contrib`), mark status as `failed_dependency` with reason code `TF1_API_INCOMPATIBLE_WITH_TF2` and stop broad migration attempts.
        15) If execution fails, still write structured outputs (`task_run_results.json`, `codex_worklog.jsonl`) for this task.

        Required execution outputs under `{outputs_dir}`:
        - Update/append `task_run_results.json` with shape:
          {{
            "runs": [
              {{
                "task_id": str,
                "entrypoint": str,
                "command": str,
                "exit_code": int,
                "status": "ok" | "failed" | "failed_dependency" | "timeout",
                "runtime_sec": float,
                "stdout_tail": str,
                "stderr_tail": str,
                "metrics": object,
                "artifacts": [str],
                "reason_codes": [str]
              }}
            ],
            "reason_codes": [str]
          }}
        - Append live command output to `{outputs_dir}/codex_exec.log`.
        - Append structured progress events to `{outputs_dir}/codex_worklog.jsonl`.
        - Write dependency actions to `{outputs_dir}/dependency_solver.json` and `{outputs_dir}/pip_install.log` when pip is used.
        - Write patch diff to `{outputs_dir}/patches.diff` if code was modified, else keep it empty.

        Keep outputs machine-readable and concise.
        """
    ).strip()


def build_codex_single_task_repair_prompt(*, outputs_dir: str, task_id: str, task_spec_path: str) -> str:
    return dedent(
        f"""
        Repair-only mode for a single task.
        Do NOT rerun full training or unrelated tasks.
        Only use `{task_spec_path}` and existing logs/artifacts to repair output structure for task_id=`{task_id}`.
        Keep other task records intact; only upsert the record for task_id=`{task_id}`.
        Only touch these files under `{outputs_dir}`:
        - `task_run_results.json`
        - `codex_worklog.jsonl`
        - `dependency_solver.json`
        - `patches.diff`
        """
    ).strip()


def build_codex_main_prompt(
    *,
    max_self_heal_iters: int | None = None,
    repo_dir: str,
    inputs_task_spec: str,
    inputs_claims_ir: str | None = None,
    inputs_metric_contract: str | None = None,
    outputs_dir: str,
) -> str:
    # Backward-compatible wrapper used by existing tests/legacy callers.
    return build_codex_single_task_prompt(
        repo_dir=repo_dir,
        inputs_task_spec=inputs_task_spec,
        inputs_metric_contract=inputs_metric_contract,
        outputs_dir=outputs_dir,
        task_id="task_single",
    )


def build_codex_repair_prompt(outputs_dir: str) -> str:
    # Backward-compatible wrapper used by existing tests/legacy callers.
    return dedent(
        f"""
        Repair-only mode.
        Do NOT rerun full training.
        Use existing logs/artifacts to repair only these files under `{outputs_dir}`:
        - `{outputs_dir}/run_manifest.json`
        - `task_run_results.json`
        - `codex_worklog.jsonl`
        - `dependency_solver.json`
        - `patches.diff`
        """
    ).strip()
