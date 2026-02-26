#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def replace_images(markdown: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        alt = (match.group("alt") or "").strip() or "(no-alt)"
        src = match.group("src").strip()
        return (
            "\n[ImageDescription]\n"
            f"- source: {src}\n"
            f"- alt: {alt}\n"
            "- description: Image found in markdown. Detailed vision caption is unavailable in this runtime.\n"
        )

    return IMAGE_RE.sub(_sub, markdown)


def convert(input_md: Path, output_md: Path) -> None:
    content = input_md.read_text(encoding="utf-8")
    converted = replace_images(content)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(converted, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace markdown images with text descriptions.")
    parser.add_argument("--input", required=True, help="Input markdown path")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
