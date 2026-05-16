"""Repository environment-file discovery helpers."""

from __future__ import annotations

import fnmatch
from pathlib import Path


CONDA_ENVIRONMENT_FILENAMES = (
    "environment.yml",
    "environment.yaml",
    "conda_env.yml",
    "conda_env.yaml",
    "conda-environment.yml",
    "conda-environment.yaml",
    "conda_environment.yml",
    "conda_environment.yaml",
    "conda.yml",
    "conda.yaml",
    "env.yml",
    "env.yaml",
    "mamba.yml",
    "mamba.yaml",
)

CONDA_ENVIRONMENT_PATTERNS = (
    "environment-*.yml",
    "environment-*.yaml",
    "environment_*.yml",
    "environment_*.yaml",
    "conda-*.yml",
    "conda-*.yaml",
    "conda_*.yml",
    "conda_*.yaml",
    "mamba-*.yml",
    "mamba-*.yaml",
    "mamba_*.yml",
    "mamba_*.yaml",
)

DEFAULT_ENV_SCAN_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


def conda_environment_file_priority(path: Path) -> int | None:
    """Return a stable priority for likely native conda/mamba env files."""
    name = path.name.lower()
    if name in CONDA_ENVIRONMENT_FILENAMES:
        return CONDA_ENVIRONMENT_FILENAMES.index(name)
    for idx, pattern in enumerate(CONDA_ENVIRONMENT_PATTERNS):
        if fnmatch.fnmatch(name, pattern):
            return len(CONDA_ENVIRONMENT_FILENAMES) + idx
    return None


def is_conda_environment_file(path: Path) -> bool:
    return conda_environment_file_priority(path) is not None


def iter_conda_environment_files(
    repo_dir: Path,
    *,
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    """Find likely conda/mamba environment files, preferring canonical names."""
    root = repo_dir.resolve()
    excluded = exclude_dirs if exclude_dirs is not None else DEFAULT_ENV_SCAN_EXCLUDE_DIRS
    candidates: list[tuple[int, int, str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in excluded for part in rel.parts):
            continue
        priority = conda_environment_file_priority(path)
        if priority is None:
            continue
        candidates.append((priority, len(rel.parts), rel.as_posix(), path))
    return [path for _, _, _, path in sorted(candidates)]
