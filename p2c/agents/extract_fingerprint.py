from __future__ import annotations

from p2c.agents.base import BaseAgent
from p2c.agents.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent
from p2c.agents.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.extract_fingerprint_guide import ExtractFingerprintGuideAgent


class ExtractFingerprintAgent(BaseAgent):
    """Compatibility wrapper that executes the 3-stage fingerprint pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(name="extract_fingerprint", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        guide = ExtractFingerprintGuideAgent(
            llm=self.llm,
            artifacts=self.artifacts,
            step_index=self.step_index,
            step_total=self.step_total,
        )
        atomic = ExtractFingerprintAtomicAgent(
            llm=self.llm,
            artifacts=self.artifacts,
            step_index=self.step_index,
            step_total=self.step_total,
        )
        filt = ExtractFingerprintFilterAgent(
            llm=self.llm,
            artifacts=self.artifacts,
            step_index=self.step_index,
            step_total=self.step_total,
        )
        guide.run(ctx)
        atomic.run(ctx)
        out = filt.run(ctx)
        return out
