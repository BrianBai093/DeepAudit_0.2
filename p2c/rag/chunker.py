"""Code chunker — splits target repo files into semantically meaningful chunks."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from p2c.agents.phase1.repo_analysis import _EXCLUDE_DIRS, _is_excluded

# File extensions to index
_INDEXABLE_EXTENSIONS = (
    ".py", ".sh", ".bash", ".yaml", ".yml", ".toml", ".cfg",
    ".ipynb", ".r", ".R", ".jl", ".lua",
)

# Also index these exact filenames regardless of extension
_INDEXABLE_NAMES = {"Makefile", "Dockerfile", "requirements.txt"}


@dataclass
class CodeChunk:
    """A single chunk of source code with provenance metadata."""

    file_path: str       # relative to repo root
    start_line: int
    end_line: int
    chunk_type: str      # "function" | "class" | "module_level" | "cell" | "fixed"
    name: str | None     # function/class name if applicable
    content: str


def chunk_repo(
    repo_dir: Path,
    *,
    max_file_size: int = 256_000,
) -> list[CodeChunk]:
    """Walk the repo tree and return code chunks for all indexable files."""
    repo_dir = repo_dir.resolve()
    chunks: list[CodeChunk] = []

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path, repo_dir):
            continue
        if path.name not in _INDEXABLE_NAMES and path.suffix not in _INDEXABLE_EXTENSIONS:
            continue
        if path.stat().st_size > max_file_size:
            continue

        rel = path.relative_to(repo_dir).as_posix()

        try:
            if path.suffix == ".py":
                chunks.extend(_chunk_python_file(path, rel))
            elif path.suffix == ".ipynb":
                chunks.extend(_chunk_notebook(path, rel))
            else:
                chunks.extend(_chunk_fixed(path, rel))
        except Exception:  # noqa: BLE001
            # If chunking fails for one file, skip it
            continue

    return chunks


# ---------------------------------------------------------------------------
# Python: AST-based chunking
# ---------------------------------------------------------------------------

def _chunk_python_file(path: Path, rel_path: str) -> list[CodeChunk]:
    """Split a .py file by top-level functions and classes."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return _chunk_fixed(path, rel_path)

    chunks: list[CodeChunk] = []
    covered: set[int] = set()  # 1-based line numbers covered by a node

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno  # 1-based
        end = node.end_lineno or start
        content = "".join(lines[start - 1 : end])
        if content.strip():
            chunks.append(CodeChunk(
                file_path=rel_path,
                start_line=start,
                end_line=end,
                chunk_type="class" if isinstance(node, ast.ClassDef) else "function",
                name=node.name,
                content=content,
            ))
            covered.update(range(start, end + 1))

    # Collect remaining module-level code as one chunk
    remaining = []
    for i, line in enumerate(lines, start=1):
        if i not in covered and line.strip():
            remaining.append((i, line))

    if remaining:
        content = "".join(ln for _, ln in remaining)
        if content.strip():
            chunks.append(CodeChunk(
                file_path=rel_path,
                start_line=remaining[0][0],
                end_line=remaining[-1][0],
                chunk_type="module_level",
                name=None,
                content=content,
            ))

    # If AST produced nothing useful, fall back to fixed chunking
    if not chunks:
        return _chunk_fixed(path, rel_path)

    return chunks


# ---------------------------------------------------------------------------
# Jupyter notebook: cell-based chunking
# ---------------------------------------------------------------------------

def _chunk_notebook(path: Path, rel_path: str) -> list[CodeChunk]:
    """Extract code cells from a .ipynb file."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    nb = json.loads(raw)
    cells = nb.get("cells", [])
    chunks: list[CodeChunk] = []
    line_cursor = 1

    for i, cell in enumerate(cells):
        if cell.get("cell_type") not in ("code", "markdown"):
            continue
        source_lines = cell.get("source", [])
        if isinstance(source_lines, str):
            source_lines = source_lines.splitlines(keepends=True)
        content = "".join(source_lines)
        if not content.strip():
            line_cursor += len(source_lines)
            continue

        n_lines = len(source_lines)
        chunks.append(CodeChunk(
            file_path=rel_path,
            start_line=line_cursor,
            end_line=line_cursor + n_lines - 1,
            chunk_type="cell",
            name=f"cell_{i}",
            content=content,
        ))
        line_cursor += n_lines

    return chunks


# ---------------------------------------------------------------------------
# Generic: fixed-size sliding window
# ---------------------------------------------------------------------------

def _chunk_fixed(
    path: Path,
    rel_path: str,
    window: int = 80,
    overlap: int = 20,
) -> list[CodeChunk]:
    """Chunk a file with a sliding window of *window* lines and *overlap* overlap."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = max(1, window - overlap)

    for start_idx in range(0, len(lines), step):
        end_idx = min(start_idx + window, len(lines))
        content = "".join(lines[start_idx:end_idx])
        if not content.strip():
            continue
        chunks.append(CodeChunk(
            file_path=rel_path,
            start_line=start_idx + 1,
            end_line=end_idx,
            chunk_type="fixed",
            name=None,
            content=content,
        ))
        if end_idx >= len(lines):
            break

    return chunks
