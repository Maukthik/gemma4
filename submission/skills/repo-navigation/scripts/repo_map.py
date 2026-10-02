#!/usr/bin/env python3
"""Print a compact map of a Python package: module -> top-level symbols."""
import argparse
import ast
import os
import sys

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".tox", ".venv", "venv", "build", "dist", "docs", ".eggs"}


def iter_py(root, include_tests):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
                             and (include_tests or d not in ("tests", "test")))
        for f in sorted(filenames):
            if f.endswith(".py") and (include_tests or not (f.startswith("test_") or f.endswith("_test.py"))):
                yield os.path.join(dirpath, f)


def summarize(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (SyntaxError, ValueError):
        return "<unparseable>"
    parts = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and (not n.name.startswith("_") or n.name in ("__init__", "__call__"))]
            m = ",".join(methods[:8]) + (",…" if len(methods) > 8 else "")
            parts.append(f"class {node.name}({m})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(("async " if isinstance(node, ast.AsyncFunctionDef) else "") + node.name + "()")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace" if os.path.isdir("/workspace") else ".")
    ap.add_argument("--max-lines", type=int, default=60)
    ap.add_argument("--include-tests", action="store_true")
    a = ap.parse_args()
    root = a.root
    if not os.path.isabs(root) and os.path.isdir("/workspace"):
        root = os.path.join("/workspace", root)  # skill scripts may not run with cwd=/workspace
    lines = []
    for p in iter_py(root, a.include_tests):
        rel = os.path.relpath(p, root)
        s = summarize(p)
        lines.append(f"{rel}: {s}" if s else rel)
    for line in lines[: a.max_lines]:
        print(line[:200])
    if len(lines) > a.max_lines:
        print(f"... {len(lines) - a.max_lines} more modules (narrow with --root)")


if __name__ == "__main__":
    sys.exit(main())
