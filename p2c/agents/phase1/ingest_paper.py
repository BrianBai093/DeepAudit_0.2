from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.PictureToWords import convert
from p2c.agents.base import BaseAgent
from p2c.utils.mineru_client import convert_pdf_to_markdown, should_generate_markdown


class IngestPaperAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="ingest_paper", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        input_path = Path(ctx["paper_md"])
        output_path = Path(ctx["paper_md_out"])
        pdf_value = ctx.get("paper_pdf")

        if pdf_value:
            pdf_path = Path(pdf_value)
            if should_generate_markdown(pdf_path, input_path):
                self.log("PROGRESS", f"generating paper markdown with MinerU: {pdf_path} -> {input_path}")
                result = convert_pdf_to_markdown(pdf_path, input_path)
                self.artifacts.write_json("paper/mineru_conversion.json", result.to_json())
                ctx["paper_md"] = str(input_path)
            else:
                self.artifacts.write_json(
                    "paper/mineru_conversion.json",
                    {
                        "provider": "cache",
                        "source_pdf": str(pdf_path),
                        "output_md": str(input_path),
                        "status": "skipped",
                        "reason": "existing markdown is up to date",
                    },
                )

        if not input_path.exists() or input_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Paper markdown not found: {input_path}. "
                "Provide --paper_md or run phase 1 with --paper_pdf so MinerU can create it."
            )

        self.log("PROGRESS", f"converting markdown images to text: {input_path} -> {output_path}")
        convert(input_md=input_path, output_md=output_path)

        return {
            "paper_md_out": str(output_path),
        }
