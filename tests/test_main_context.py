from __future__ import annotations

import json

from p2c.main import serializable_context


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
