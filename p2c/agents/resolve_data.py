from __future__ import annotations

from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.schemas import DataManifest, DataManifestEntry

SYSTEM_PROMPT = "You summarize dataset readiness in strict JSON with reason_codes."
USER_PROMPT_TEMPLATE = "Input: repo files. Output: execution/data_manifest.json"


class ResolveDataAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="resolve_data", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        repo_dir = Path(ctx["repo_dir"])
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        candidates = []
        for pattern in ("*.csv", "*.json", "*.data", "*.npy", "*.txt"):
            candidates.extend(repo_dir.rglob(pattern))

        entries: list[DataManifestEntry] = []
        for p in sorted(candidates)[:100]:
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            entries.append(
                DataManifestEntry(
                    path=str(p.relative_to(repo_dir)),
                    exists=p.exists(),
                    size_bytes=size,
                )
            )

        unresolved = len(entries) == 0
        reason_codes = ["NO_DATA_FILES_DISCOVERED"] if unresolved else []
        manifest = DataManifest(entries=entries, unresolved=unresolved, reason_codes=reason_codes)
        self.artifacts.write_json("execution/data_manifest.json", manifest.model_dump())
        return {"data_manifest": manifest.model_dump()}
