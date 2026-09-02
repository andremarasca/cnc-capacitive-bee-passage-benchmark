"""Deterministic provenance for the frozen scientific source tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


def find_project_root(start: str | Path) -> Path:
    """Find the nearest parent containing the package and project metadata."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "src" / "bee_counter_bench"
        ).is_dir():
            return directory
    raise FileNotFoundError(f"Cannot locate project root from {start}.")


def source_code_sha256(project_root: str | Path) -> str:
    """Hash every scientific source and frozen configuration file."""
    root = find_project_root(project_root)
    paths = [
        path
        for path in (root / "src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.extend(path for path in (root / "configs").glob("*.yaml") if path.is_file())
    paths.extend(
        path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file()
    )
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        content_digest = hashlib.sha256(path.read_bytes()).digest()
        hasher.update(len(relative_path).to_bytes(4, "big"))
        hasher.update(relative_path)
        hasher.update(content_digest)
    return hasher.hexdigest()
