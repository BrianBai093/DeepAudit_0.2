from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from p2c.schemas import VerdictDoc

REQUIRED_FILES = [
    "fingerprint/fingerprint.json",
    "fingerprint/guide_sentences.json",
    "fingerprint/atomic_criteria.json",
    "fingerprint/atomic_rejected.json",
    "fingerprint/filter_clusters.json",
    "fingerprint/filter_selected.json",
    "fingerprint/claims_ir.json",
    "task/task_spec.json",
    "task/metric_contract.json",
    "execution/run.log",
    "execution/commands.jsonl",
    "execution/patch.diff",
    "execution/codex_outputs/run_manifest.json",
    "execution/codex_outputs/codex_worklog.jsonl",
    "execution/codex_outputs/patches.diff",
    "execution/codex_outputs/claim_alignment.json",
    "execution/codex_outputs/codex_exec.log",
    "execution/codex_outputs/dependency_solver.json",
    "execution/codex_outputs/toolchain_probe.json",
    "execution/codex_outputs/pip_install.log",
    "execution/codex_outputs/capability_probe.json",
    "execution/codex_outputs/dependency_bootstrap.log",
    "execution/codex_outputs/codex_failure.json",
    "execution/codex_outputs/codex_main.log",
    "execution/codex_outputs/codex_repair.log",
    "execution/codex_outputs/task_run_results.json",
    "execution/codex_outputs/codex_exec.stream.log",
    "execution/repo_state.json",
    "execution/codex_failure.json",
    "execution/system_info.json",
    "execution/env_lock/pip_freeze.txt",
    "execution/data_manifest.json",
    "results/metrics.json",
    "results/parsed_evidence.json",
    "results/evaluability.json",
    "results/evaluability_verdict.json",
    "results/verdict.json",
    "results/report.md",
]


class ArtifactManager:
    def __init__(self, artifacts_dir: str | Path, run_id: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.run_id = run_id
        self.run_root = self.artifacts_dir / run_id

    def ensure_tree(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for rel in REQUIRED_FILES:
            path = self.run_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue
            if path.suffix == ".json":
                if rel.endswith("verdict.json"):
                    payload = VerdictDoc(
                        status="INCONCLUSIVE",
                        reason_codes=["INITIALIZED_PLACEHOLDER"],
                        summary="Pipeline not complete yet.",
                    ).model_dump()
                elif rel.endswith("metrics.json"):
                    payload = {"records": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("parsed_evidence.json"):
                    payload = {"claim_evidence": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("run_manifest.json"):
                    payload = {"runs": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("claim_alignment.json"):
                    payload = {"claims": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("codex_failure.json"):
                    payload = {
                        "stage": "postcheck",
                        "last_command": "",
                        "exit_code": 0,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "codex_exec_log_tail": "",
                        "pip_log_tail": "",
                        "capability_snapshot": {},
                        "dependency_bootstrap_trace": [],
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("capability_probe.json"):
                    payload = {
                        "python_ok": False,
                        "python_version": "",
                        "pip_available": False,
                        "ensurepip_available": False,
                        "required_modules_available": {},
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("dependency_solver.json"):
                    payload = {
                        "steps": [],
                        "status": "not_run",
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("toolchain_probe.json"):
                    payload = {
                        "paths": {},
                        "versions": {},
                        "path_prefix": "",
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("task_run_results.json"):
                    payload = {"runs": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("evaluability.json"):
                    payload = {"entries": [], "reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                elif rel.endswith("evaluability_verdict.json"):
                    payload = {
                        "status": "NOT_EVALUABLE",
                        "claim_rows": [],
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                        "summary": "Pipeline not complete yet.",
                    }
                elif rel.endswith("task_spec.json"):
                    payload = {
                        "tasks": [],
                        "constraints": {},
                        "entrypoints": [],
                        "metric_observers": [],
                        "run_matrix": [],
                        "selection_notes": [],
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("metric_contract.json"):
                    payload = {
                        "required_metrics": [],
                        "parsers": [],
                        "normalization": {},
                        "reason_codes": ["INITIALIZED_PLACEHOLDER"],
                    }
                elif rel.endswith("repo_state.json"):
                    payload = {
                        "head": None,
                        "branch": None,
                        "diff_summary": None,
                        "submodules": [],
                        "reason_codes": ["INITIALIZED_PLACEHOLDER", "NO_GIT_METADATA"],
                    }
                else:
                    payload = {"reason_codes": ["INITIALIZED_PLACEHOLDER"]}
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                path.write_text("", encoding="utf-8")

    def path(self, relative: str) -> Path:
        return self.run_root / relative

    def write_json(self, relative: str, payload: Any) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        self._atomic_write(path, content)
        return path

    def append_jsonl(self, relative: str, record: dict[str, Any]) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def write_text(self, relative: str, content: str) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, content)
        return path

    def append_text(self, relative: str, content: str) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return path

    def read_json(self, relative: str) -> dict[str, Any]:
        path = self.path(relative)
        if not path.exists() or path.stat().st_size == 0:
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def sha256_file(self, relative: str) -> str:
        path = self.path(relative)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8192):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
