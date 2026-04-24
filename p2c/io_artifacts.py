from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from p2c.schemas import VerdictDoc

REQUIRED_FILES = [
    # Phase 1 — fingerprint
    "fingerprint/fingerprint.json",
    "fingerprint/guide_sentences.json",
    "fingerprint/atomic_criteria.json",
    "fingerprint/atomic_rejected.json",
    "fingerprint/filter_clusters.json",
    "fingerprint/filter_selected.json",
    "fingerprint/claims_ir.json",
    "fingerprint/visual_targets.json",
    # Phase 1 — task compilation
    "task/repo_analysis.json",
    "task/task_spec.json",
    "task/metric_contract.json",
    # Phase 2 — local execution
    "execution/run.log",
    "execution/env_setup_result.json",
    "execution/execution_failures.json",
    "execution/phase2_state.json",
    "execution/executor_outputs/run_manifest.json",
    "execution/executor_outputs/executor_agent.log",
    "execution/executor_outputs/executor_activity.jsonl",
    "execution/executor_outputs/executor_runtime.json",
    "execution/executor_outputs/session_stdout.log",
    "execution/executor_outputs/session_stderr.log",
    "execution/env_lock/pip_freeze.txt",
    # Phase 3 — verification
    "results/metrics.json",
    "results/parsed_evidence.json",
    "results/evaluability.json",
    "results/evaluability_verdict.json",
    "results/verdict.json",
    "results/visual_to_repo_alignment.json",
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
                payload = self._default_json_payload(rel)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                path.write_text("", encoding="utf-8")

    @staticmethod
    def _default_json_payload(rel: str) -> Any:
        """Return a sensible placeholder JSON payload for a given artifact path."""
        _P = "INITIALIZED_PLACEHOLDER"
        _MAP: dict[str, Any] = {
            "verdict.json": {"status": "INCONCLUSIVE", "claim_verdicts": [],
                             "reason_codes": [_P], "summary": "Pipeline not complete yet."},
            "metrics.json": {"records": [], "reason_codes": [_P]},
            "parsed_evidence.json": {"claim_evidence": [], "reason_codes": [_P]},
            "run_manifest.json": {"runs": [], "reason_codes": [_P]},
            "evaluability.json": {"entries": [], "reason_codes": [_P]},
            "evaluability_verdict.json": {"status": "NOT_EVALUABLE", "claim_rows": [],
                                          "reason_codes": [_P], "summary": "Pipeline not complete yet."},
            "visual_to_repo_alignment.json": {"alignments": [], "reason_codes": [_P]},
            "task_spec.json": {"tasks": [], "constraints": {}, "entrypoints": [],
                               "metric_observers": [], "run_matrix": [], "selection_notes": [],
                               "reason_codes": [_P]},
            "repo_analysis.json": {"ecosystems": [], "dependency_profiles": [],
                                   "entrypoint_candidates": [], "primary_entrypoint_id": None,
                                   "reason_codes": [_P]},
            "visual_targets.json": {"visual_targets": [], "reason_codes": [_P]},
            "metric_contract.json": {"required_metrics": [], "parsers": [], "normalization": {},
                                     "reason_codes": [_P]},
            "env_setup_result.json": {"env_name": "", "validation_passed": False,
                                      "reason_codes": [_P]},
            "execution_failures.json": [],
            "phase2_state.json": {"status": "env_setup", "attempt": 0, "failures": [],
                                  "reason_codes": [_P]},
        }
        for suffix, payload in _MAP.items():
            if rel.endswith(suffix):
                return payload
        return {"reason_codes": [_P]}

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
