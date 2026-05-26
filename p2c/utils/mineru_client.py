from __future__ import annotations

import io
import json
import os
import tempfile
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LIGHTWEIGHT_MAX_BYTES = 10 * 1024 * 1024
MINERU_AGENT_BASE_URL = "https://mineru.net/api/v1/agent"
MINERU_STANDARD_BASE_URL = "https://mineru.net/api/v4"
LIGHTWEIGHT_LIMIT_CODES = {-30001, -30002, -30003}


class MinerUError(RuntimeError):
    pass


class MinerULightweightLimitError(MinerUError):
    pass


@dataclass
class MinerUConversionResult:
    provider: str
    source_pdf: str
    output_md: str
    status: str
    task_id: str | None = None
    batch_id: str | None = None
    markdown_url: str | None = None
    full_zip_url: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class MinerUHTTPTransport:
    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = dict(headers or {})
        if payload is not None:
            request_headers.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body, method=method, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise MinerUError(f"MinerU HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise MinerUError(f"MinerU request failed: {e}") from e
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise MinerUError(f"MinerU returned non-JSON response from {url}: {raw[:300]}") from e

    def put_file(self, url: str, path: Path, timeout: int = 300) -> int:
        # MinerU/OSS signed upload URLs are sensitive to signed headers. urllib
        # automatically adds Content-Type: application/x-www-form-urlencoded
        # when data is present, which changes OSS's StringToSign and causes
        # SignatureDoesNotMatch. Use http.client so the PUT carries no
        # Content-Type header unless MinerU signs one in the future.
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise MinerUError(f"Invalid MinerU upload URL: {url}")
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        data = path.read_bytes()
        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = connection_cls(parsed.netloc, timeout=timeout)
        try:
            conn.request(
                "PUT",
                target,
                body=data,
                headers={"Content-Length": str(len(data))},
            )
            resp = conn.getresponse()
            details = resp.read().decode("utf-8", errors="ignore")
            if resp.status >= 400:
                raise MinerUError(f"MinerU upload failed with HTTP {resp.status}: {details}")
            return int(resp.status)
        except OSError as e:
            raise MinerUError(f"MinerU upload failed: {e}") from e
        finally:
            conn.close()

    def get_text(self, url: str, timeout: int = 120) -> str:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise MinerUError(f"MinerU download failed with HTTP {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise MinerUError(f"MinerU download failed: {e}") from e

    def get_bytes(self, url: str, timeout: int = 300) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise MinerUError(f"MinerU download failed with HTTP {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise MinerUError(f"MinerU download failed: {e}") from e


def default_paper_md_from_pdf(paper_pdf: str | Path) -> Path:
    pdf_path = Path(paper_pdf)
    if pdf_path.parent.name == "paper":
        return pdf_path.parent / "full.md"
    return pdf_path.parent / "paper" / "full.md"


def should_generate_markdown(pdf_path: Path, md_path: Path, force: bool | None = None) -> bool:
    if force is None:
        force = _env_bool("P2C_MINERU_FORCE", False)
    if force:
        return True
    if not md_path.exists() or md_path.stat().st_size == 0:
        return True
    if not pdf_path.exists():
        return False
    return pdf_path.stat().st_mtime > md_path.stat().st_mtime


def convert_pdf_to_markdown(
    pdf_path: Path,
    md_path: Path,
    transport: MinerUHTTPTransport | None = None,
) -> MinerUConversionResult:
    pdf_path = Path(pdf_path)
    md_path = Path(md_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Paper PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise MinerUError(f"MinerU conversion expects a PDF input: {pdf_path}")

    transport = transport or MinerUHTTPTransport()
    mode = os.getenv("P2C_MINERU_MODE", "auto").strip().lower() or "auto"
    token = (os.getenv("MINERU_API_TOKEN") or "").strip()

    if mode in {"off", "disabled", "none"}:
        raise MinerUError("P2C_MINERU_MODE disables automatic PDF to Markdown conversion")
    if mode not in {"auto", "agent", "lightweight", "standard", "precise"}:
        raise MinerUError(
            "Unsupported P2C_MINERU_MODE. Use auto, agent, lightweight, standard, precise, or off."
        )

    if mode in {"standard", "precise"}:
        return _convert_standard(pdf_path, md_path, token=token, transport=transport)

    if mode == "auto" and pdf_path.stat().st_size > LIGHTWEIGHT_MAX_BYTES:
        if not token:
            raise MinerUError(
                f"{pdf_path} is larger than the MinerU lightweight 10MB limit. "
                "Set MINERU_API_TOKEN to use the standard MinerU API."
            )
        return _convert_standard(pdf_path, md_path, token=token, transport=transport)

    try:
        return _convert_lightweight(pdf_path, md_path, transport=transport)
    except MinerULightweightLimitError:
        if mode in {"agent", "lightweight"} or not token:
            raise
        return _convert_standard(pdf_path, md_path, token=token, transport=transport)


def _convert_lightweight(
    pdf_path: Path,
    md_path: Path,
    transport: MinerUHTTPTransport,
) -> MinerUConversionResult:
    payload = _common_parse_payload(pdf_path.name, page_key="page_range")
    payload.pop("_file_options", None)
    response = transport.request_json("POST", f"{MINERU_AGENT_BASE_URL}/parse/file", payload=payload)
    _raise_for_api_error(response, "create MinerU lightweight upload task")
    data = _require_dict(response, "data")
    task_id = _require_str(data, "task_id")
    file_url = _require_str(data, "file_url")

    status = transport.put_file(file_url, pdf_path)
    if status not in {200, 201}:
        raise MinerUError(f"MinerU lightweight upload failed with HTTP status {status}")

    result_data = _poll_lightweight(task_id, transport)
    markdown_url = _require_str(result_data, "markdown_url")
    markdown = transport.get_text(markdown_url)
    _atomic_write_text(md_path, markdown)
    return MinerUConversionResult(
        provider="mineru_agent",
        source_pdf=str(pdf_path),
        output_md=str(md_path),
        status="done",
        task_id=task_id,
        markdown_url=markdown_url,
    )


def _convert_standard(
    pdf_path: Path,
    md_path: Path,
    token: str,
    transport: MinerUHTTPTransport,
) -> MinerUConversionResult:
    if not token:
        raise MinerUError("MINERU_API_TOKEN is required for the standard MinerU API")

    data_id = _data_id_for(pdf_path)
    payload = _common_parse_payload(pdf_path.name, page_key="page_ranges")
    payload["files"] = [{"name": pdf_path.name, "data_id": data_id, **payload.pop("_file_options")}]
    payload.pop("file_name", None)
    payload.pop("is_ocr", None)
    payload.pop("page_ranges", None)
    payload["model_version"] = os.getenv("P2C_MINERU_MODEL_VERSION", "vlm").strip() or "vlm"
    headers = _standard_headers(token)

    response = transport.request_json(
        "POST",
        f"{MINERU_STANDARD_BASE_URL}/file-urls/batch",
        payload=payload,
        headers=headers,
    )
    _raise_for_api_error(response, "create MinerU standard upload task")
    data = _require_dict(response, "data")
    batch_id = _require_str(data, "batch_id")
    file_urls = data.get("file_urls")
    if not isinstance(file_urls, list) or not file_urls:
        raise MinerUError(f"MinerU standard response missing file_urls: {response}")

    status = transport.put_file(str(file_urls[0]), pdf_path)
    if status not in {200, 201}:
        raise MinerUError(f"MinerU standard upload failed with HTTP status {status}")

    result = _poll_standard(batch_id, pdf_path.name, data_id, token, transport)
    full_zip_url = _require_str(result, "full_zip_url")
    archive = transport.get_bytes(full_zip_url)
    markdown = _extract_full_markdown(archive)
    _atomic_write_text(md_path, markdown)
    return MinerUConversionResult(
        provider="mineru_standard",
        source_pdf=str(pdf_path),
        output_md=str(md_path),
        status="done",
        batch_id=batch_id,
        full_zip_url=full_zip_url,
    )


def _common_parse_payload(file_name: str, page_key: str) -> dict[str, Any]:
    language = os.getenv("P2C_MINERU_LANGUAGE", "en").strip() or "en"
    page_range = os.getenv("P2C_MINERU_PAGE_RANGE", "").strip()
    is_ocr = _env_bool("P2C_MINERU_OCR", False)
    enable_table = _env_bool("P2C_MINERU_ENABLE_TABLE", True)
    enable_formula = _env_bool("P2C_MINERU_ENABLE_FORMULA", True)
    payload: dict[str, Any] = {
        "file_name": file_name,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
        "_file_options": {"is_ocr": is_ocr},
    }
    if page_range:
        payload[page_key] = page_range
        payload["_file_options"][page_key] = page_range
    return payload


def _poll_lightweight(task_id: str, transport: MinerUHTTPTransport) -> dict[str, Any]:
    deadline = time.time() + _env_int("P2C_MINERU_TIMEOUT_SEC", 900)
    interval = _env_int("P2C_MINERU_POLL_INTERVAL_SEC", 3)
    last_state = ""
    while time.time() < deadline:
        response = transport.request_json("GET", f"{MINERU_AGENT_BASE_URL}/parse/{task_id}")
        _raise_for_api_error(response, "poll MinerU lightweight task")
        data = _require_dict(response, "data")
        state = str(data.get("state") or "")
        last_state = state
        if state == "done":
            return data
        if state == "failed":
            err_code = data.get("err_code")
            err_msg = str(data.get("err_msg") or "MinerU lightweight task failed")
            if err_code in LIGHTWEIGHT_LIMIT_CODES or "lightweight API limit" in err_msg:
                raise MinerULightweightLimitError(err_msg)
            raise MinerUError(f"MinerU lightweight task failed: {err_code} {err_msg}")
        time.sleep(interval)
    raise MinerUError(f"Timed out waiting for MinerU lightweight task {task_id}; last_state={last_state}")


def _poll_standard(
    batch_id: str,
    file_name: str,
    data_id: str,
    token: str,
    transport: MinerUHTTPTransport,
) -> dict[str, Any]:
    deadline = time.time() + _env_int("P2C_MINERU_TIMEOUT_SEC", 900)
    interval = _env_int("P2C_MINERU_POLL_INTERVAL_SEC", 3)
    headers = _standard_headers(token)
    last_state = ""
    while time.time() < deadline:
        response = transport.request_json(
            "GET",
            f"{MINERU_STANDARD_BASE_URL}/extract-results/batch/{batch_id}",
            headers=headers,
        )
        _raise_for_api_error(response, "poll MinerU standard task")
        data = _require_dict(response, "data")
        results = data.get("extract_result")
        if not isinstance(results, list):
            raise MinerUError(f"MinerU standard response missing extract_result: {response}")
        selected = _select_standard_result(results, file_name, data_id)
        state = str(selected.get("state") or "")
        last_state = state
        if state == "done":
            return selected
        if state == "failed":
            raise MinerUError(f"MinerU standard task failed: {selected.get('err_msg') or selected}")
        time.sleep(interval)
    raise MinerUError(f"Timed out waiting for MinerU standard batch {batch_id}; last_state={last_state}")


def _select_standard_result(results: list[Any], file_name: str, data_id: str) -> dict[str, Any]:
    dicts = [item for item in results if isinstance(item, dict)]
    for item in dicts:
        if item.get("data_id") == data_id:
            return item
    for item in dicts:
        if item.get("file_name") == file_name:
            return item
    if dicts:
        return dicts[0]
    raise MinerUError("MinerU standard response contains no task results")


def _extract_full_markdown(archive: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        candidates = [
            name for name in zf.namelist()
            if name.endswith("full.md") and not name.endswith("/")
        ]
        if not candidates:
            raise MinerUError("MinerU result zip does not contain full.md")
        candidates.sort(key=lambda name: (name.count("/"), len(name), name))
        with zf.open(candidates[0]) as f:
            return f.read().decode("utf-8")


def _raise_for_api_error(response: dict[str, Any], action: str) -> None:
    code = response.get("code")
    if code == 0:
        return
    raise MinerUError(f"MinerU failed to {action}: code={code} msg={response.get('msg')}")


def _require_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise MinerUError(f"MinerU response missing object field {key}: {container}")
    return value


def _require_str(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise MinerUError(f"MinerU response missing string field {key}: {container}")
    return value


def _standard_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


def _data_id_for(pdf_path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pdf_path.stem)
    return f"p2c_{stem}"[:128]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_mineru_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
