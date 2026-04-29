from __future__ import annotations

import json

from p2c.io_artifacts import ArtifactManager
from p2c.main import ensure_phase_prereq, serializable_context


class NonSerializableRuntimeObject:
    pass


def test_serializable_context_drops_internal_runtime_objects() -> None:
    ctx = {
        "phase": 1,
        "repo_dir": "Target/code",
        "_code_index": NonSerializableRuntimeObject(),
        "_p2_state": NonSerializableRuntimeObject(),
    }

    payload = serializable_context(ctx)

    assert payload == {"phase": 1, "repo_dir": "Target/code"}
    json.dumps(payload)


def test_phase3_prereq_accepts_executor_results_when_manifest_is_placeholder(tmp_path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run")
    artifacts.ensure_tree()
    artifacts.write_json("execution/executor_outputs/run_manifest.json", {"runs": [], "reason_codes": []})
    artifacts.write_json(
        "execution/executor_outputs/executor_results.json",
        {"runs": [{"experiment_id": "exp_01", "command": "python train.py", "status": "ok"}]},
    )

    ensure_phase_prereq(3, artifacts)


def test_phase3_prereq_accepts_phase2_execution_package(tmp_path) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run")
    artifacts.ensure_tree()
    artifacts.write_json("execution/executor_outputs/run_manifest.json", {"runs": [], "reason_codes": []})
    artifacts.write_json(
        "execution/executor_outputs/phase2_execution_package.json",
        {
            "schema_version": "phase2_execution_package.v1",
            "experiments": [{"experiment_id": "exp_01", "attempts": []}],
            "reason_codes": [],
        },
    )

    ensure_phase_prereq(3, artifacts)
