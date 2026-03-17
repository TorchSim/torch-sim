"""Check whether a single model test needs to run given the changed files.

Combines internal repo import closure with external package scanning
(the external package must already be installed in the current env).

Exit codes:
  0 — model is affected, run the tests
  1 — model is NOT affected, skip the tests
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_PREFIXES = ("torch_sim/", "tests/")
GLOBAL_TRIGGER_PATHS = {
    ".github/scripts/check_model_affected.py",
    ".github/workflows/model-tests.json",
    ".github/workflows/test.yml",
    "pyproject.toml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument(
        "--external-package",
        default=None,
        help="External package name to scan for torch_sim imports "
        "(must already be installed).",
    )
    return parser.parse_args()


def _relative_posix(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _discover_python_files() -> set[str]:
    python_files: set[str] = set()
    for prefix in ("torch_sim", "tests"):
        for path in (REPO_ROOT / prefix).rglob("*.py"):
            python_files.add(_relative_posix(path))
    return python_files


def _module_name_for_path(path: str) -> str:
    pure = PurePosixPath(path)
    parts = list(pure.parts)
    if pure.name == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = pure.stem
    return ".".join(parts)


def _build_module_index(python_files: set[str]) -> dict[str, str]:
    return {_module_name_for_path(p): p for p in python_files}


def _current_package(module_name: str, path: str) -> str:
    if PurePosixPath(path).name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def _resolve_relative_module(
    module_name: str | None, level: int, package_name: str
) -> str | None:
    if level == 0:
        return module_name
    package_parts = package_name.split(".")
    ascents = level - 1
    if ascents > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascents]
    if module_name:
        return ".".join([*base_parts, *module_name.split(".")])
    return ".".join(base_parts)


def _resolve_import_targets(path: str, module_index: dict[str, str]) -> set[str]:  # noqa: C901
    source = (REPO_ROOT / path).read_text()
    tree = ast.parse(source, filename=path)
    module_name = _module_name_for_path(path)
    package_name = _current_package(module_name, path)
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                t = module_index.get(alias.name)
                if t is not None:
                    targets.add(t)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative_module(node.module, node.level, package_name)
            if base is None:
                continue
            bt = module_index.get(base)
            if bt is not None:
                targets.add(bt)
            for alias in node.names:
                if alias.name == "*":
                    continue
                cand = f"{base}.{alias.name}" if base else alias.name
                t = module_index.get(cand)
                if t is not None:
                    targets.add(t)
    return targets


def _build_import_graph(
    python_files: set[str], module_index: dict[str, str]
) -> dict[str, set[str]]:
    return {p: _resolve_import_targets(p, module_index) for p in python_files}


def _pytest_conftest_paths(path: str) -> set[str]:
    conftests: set[str] = set()
    current = PurePosixPath(path).parent
    while str(current) not in ("", "."):
        candidate = current / "conftest.py"
        if (REPO_ROOT / candidate).is_file():
            conftests.add(candidate.as_posix())
        current = current.parent
    return conftests


def _transitive_closure(graph: dict[str, set[str]], roots: set[str]) -> set[str]:
    visited: set[str] = set()
    queue = deque(roots)
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        queue.extend(graph.get(path, ()))
    return visited


def _changed_files(base: str, head: str) -> set[str]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    result = subprocess.run(  # noqa: S603
        [git, "diff", "--name-only", base, head],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def _scan_external_package(package_name: str, module_index: dict[str, str]) -> set[str]:
    """Scan an already-installed external package for torch_sim imports."""
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return set()
    origin = Path(spec.origin)
    pkg_root = origin.parent if origin.name == "__init__.py" else origin.parent  # noqa: RUF034

    repo_files: set[str] = set()
    for py_file in pkg_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("torch_sim"):
                        _resolve_module_to_file(alias.name, module_index, repo_files)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("torch_sim"):
                    _resolve_module_to_file(mod, module_index, repo_files)
    return repo_files


def _resolve_module_to_file(
    module: str,
    module_index: dict[str, str],
    out: set[str],
) -> None:
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_index:
            out.add(module_index[candidate])
            return


def main() -> int:
    """Exit 0 if the model is affected by changed files, 1 otherwise."""
    args = _parse_args()
    repo_changes = _changed_files(args.base, args.head)

    if repo_changes & GLOBAL_TRIGGER_PATHS:
        sys.stderr.write("Global trigger path changed — running tests\n")
        return 0

    internal_changes = {p for p in repo_changes if p.startswith(INTERNAL_PREFIXES)}
    if not internal_changes:
        sys.stderr.write("No internal changes — skipping tests\n")
        return 1

    python_files = _discover_python_files()
    module_index = _build_module_index(python_files)

    unknown = {
        p for p in internal_changes if not p.endswith(".py") or p not in python_files
    }
    if unknown:
        sys.stderr.write(f"Unclassifiable internal changes {unknown} — running tests\n")
        return 0

    graph = _build_import_graph(python_files, module_index)

    roots = {args.test_path, *_pytest_conftest_paths(args.test_path)}

    if args.external_package:
        ext_files = _scan_external_package(args.external_package, module_index)
        if ext_files:
            sys.stderr.write(f"External deps on torch_sim: {sorted(ext_files)}\n")
        roots.update(ext_files)

    closure = _transitive_closure(graph, roots)
    overlap = closure & internal_changes
    if overlap:
        sys.stderr.write(f"Affected files: {sorted(overlap)}\n")
        return 0

    sys.stderr.write("No affected files in closure — skipping tests\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
