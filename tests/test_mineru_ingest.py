from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from p2c.agents.phase1.ingest_paper import IngestPaperAgent
from p2c.io_artifacts import ArtifactManager
from p2c.utils.mineru_client import (
    MinerUConversionResult,
    MinerUError,
    MinerUHTTPTransport,
    convert_pdf_to_markdown,
    default_paper_md_from_pdf,
    should_generate_markdown,
)


class DummyLLM:
    pass


class FakeMinerUTransport:
    def __init__(self, lightweight_failed: bool = False):
        self.lightweight_failed = lightweight_failed
        self.calls: list[tuple[str, str]] = []

    def request_json(self, method, url, payload=None, headers=None, timeout=60):
        self.calls.append((method, url))
        if url.endswith("/api/v1/agent/parse/file") or url.endswith("/parse/file"):
            return {"code": 0, "data": {"task_id": "task-1", "file_url": "https://upload/light"}}
        if url.endswith("/parse/task-1"):
            if self.lightweight_failed:
                return {
                    "code": 0,
                    "data": {
                        "task_id": "task-1",
                        "state": "failed",
                        "err_code": -30003,
                        "err_msg": "file page count exceeds lightweight API limit",
                    },
                }
            return {
                "code": 0,
                "data": {
                    "task_id": "task-1",
                    "state": "done",
                    "markdown_url": "https://cdn/full.md",
                },
            }
        if url.endswith("/api/v4/file-urls/batch") or url.endswith("/file-urls/batch"):
            return {"code": 0, "data": {"batch_id": "batch-1", "file_urls": ["https://upload/std"]}}
        if url.endswith("/extract-results/batch/batch-1"):
            return {
                "code": 0,
                "data": {
                    "batch_id": "batch-1",
                    "extract_result": [
                        {
                            "file_name": "paper.pdf",
                            "state": "done",
                            "full_zip_url": "https://cdn/result.zip",
                        }
                    ],
                },
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    def put_file(self, url, path, timeout=300):
        self.calls.append(("PUT", url))
        return 200

    def get_text(self, url, timeout=120):
        self.calls.append(("GET_TEXT", url))
        return "# Lightweight Markdown\n"

    def get_bytes(self, url, timeout=300):
        self.calls.append(("GET_BYTES", url))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("nested/full.md", "# Standard Markdown\n")
        return buf.getvalue()


def test_put_file_does_not_send_content_type_header(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    captured: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b""

    class FakeConnection:
        def __init__(self, netloc: str, timeout: int):
            captured["netloc"] = netloc
            captured["timeout"] = timeout

        def request(self, method: str, target: str, body: bytes, headers: dict[str, str]):
            captured["method"] = method
            captured["target"] = target
            captured["body"] = body
            captured["headers"] = headers

        def getresponse(self):
            return FakeResponse()

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("p2c.utils.mineru_client.http.client.HTTPSConnection", FakeConnection)

    status = MinerUHTTPTransport().put_file(
        "https://mineru.oss-cn-shanghai.aliyuncs.com/api-upload/file.pdf?Expires=1&Signature=abc",
        pdf,
    )

    assert status == 200
    assert captured["method"] == "PUT"
    assert captured["target"] == "/api-upload/file.pdf?Expires=1&Signature=abc"
    assert captured["body"] == b"%PDF"
    assert captured["headers"] == {"Content-Length": "4"}
    assert captured["closed"] is True


def test_default_paper_md_from_case_pdf() -> None:
    assert default_paper_md_from_pdf("paper_with_code/run_a/paper.pdf") == Path(
        "paper_with_code/run_a/paper/full.md"
    )
    assert default_paper_md_from_pdf("paper_with_code/run_a/paper/paper.pdf") == Path(
        "paper_with_code/run_a/paper/full.md"
    )


def test_should_generate_markdown_when_missing_or_pdf_newer(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    md = tmp_path / "paper" / "full.md"
    pdf.write_bytes(b"%PDF")

    assert should_generate_markdown(pdf, md)
    md.parent.mkdir()
    md.write_text("# old\n", encoding="utf-8")
    os.utime(pdf, (100, 100))
    os.utime(md, (200, 200))
    assert not should_generate_markdown(pdf, md)
    pdf.write_bytes(b"%PDF newer")
    os.utime(pdf, (300, 300))
    assert should_generate_markdown(pdf, md)


def test_convert_pdf_to_markdown_lightweight_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("P2C_MINERU_MODE", "agent")
    pdf = tmp_path / "paper.pdf"
    md = tmp_path / "paper" / "full.md"
    pdf.write_bytes(b"%PDF")

    result = convert_pdf_to_markdown(pdf, md, transport=FakeMinerUTransport())

    assert result.provider == "mineru_agent"
    assert md.read_text(encoding="utf-8") == "# Lightweight Markdown\n"


def test_convert_pdf_to_markdown_falls_back_to_standard_with_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("P2C_MINERU_MODE", "auto")
    monkeypatch.setenv("MINERU_API_TOKEN", "token")
    pdf = tmp_path / "paper.pdf"
    md = tmp_path / "paper" / "full.md"
    pdf.write_bytes(b"%PDF")
    transport = FakeMinerUTransport(lightweight_failed=True)

    result = convert_pdf_to_markdown(pdf, md, transport=transport)

    assert result.provider == "mineru_standard"
    assert md.read_text(encoding="utf-8") == "# Standard Markdown\n"
    assert any("/file-urls/batch" in url for _, url in transport.calls)


def test_convert_pdf_to_markdown_large_auto_requires_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("P2C_MINERU_MODE", "auto")
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    pdf = tmp_path / "paper.pdf"
    md = tmp_path / "paper" / "full.md"
    with pdf.open("wb") as f:
        f.truncate(11 * 1024 * 1024)

    try:
        convert_pdf_to_markdown(pdf, md, transport=FakeMinerUTransport())
    except MinerUError as exc:
        assert "MINERU_API_TOKEN" in str(exc)
    else:
        raise AssertionError("expected MinerUError for large PDF without token")


def test_ingest_generates_missing_markdown_before_picture_conversion(tmp_path: Path, monkeypatch) -> None:
    artifacts = ArtifactManager(tmp_path / "artifacts", "run")
    artifacts.ensure_tree()
    pdf = tmp_path / "case" / "paper.pdf"
    md = tmp_path / "case" / "paper" / "full.md"
    out = tmp_path / "out" / "paper.md"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")

    def fake_convert_pdf(pdf_path: Path, md_path: Path):
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("# Generated\n", encoding="utf-8")
        return MinerUConversionResult(
            provider="fake",
            source_pdf=str(pdf_path),
            output_md=str(md_path),
            status="done",
        )

    monkeypatch.setattr("p2c.agents.phase1.ingest_paper.convert_pdf_to_markdown", fake_convert_pdf)

    agent = IngestPaperAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)
    agent.execute({"paper_md": str(md), "paper_md_out": str(out), "paper_pdf": str(pdf)})

    assert md.read_text(encoding="utf-8") == "# Generated\n"
    assert out.read_text(encoding="utf-8") == "# Generated\n"
    assert artifacts.read_json("paper/mineru_conversion.json")["provider"] == "fake"
