from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.PictureToWords import convert
from p2c.agents.base import BaseAgent


class IngestPaperAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="ingest_paper", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        input_path = Path(ctx["paper_md"])
        output_path = Path(ctx["paper_md_out"])

        self.log("PROGRESS", f"converting markdown images to text: {input_path} -> {output_path}")
        convert(input_md=input_path, output_md=output_path)

        return {
            "paper_md_out": str(output_path),
        }
