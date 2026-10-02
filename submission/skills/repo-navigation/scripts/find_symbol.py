#!/usr/bin/env python3
"""Find definitions and usages of a Python identifier (attribute paths allowed: Class.method)."""
import argparse
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", ".tox", ".venv", "venv", "build", "dist", ".eggs", "node_modules"}


def is_test(path):
    parts = path.replace("\\", "/").split("/")
    base = parts[-1]
    return any(p in ("tests", "test", "testing") for p in parts[:-1]) or base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--max-usages", type=int, default=40)
    a = ap.parse_args()
    root = a.root
    if not os.path.isdir(root) and os.path.isdir("/workspace"):
        root = "/workspace"
    leaf = a.name.split(".")[-1]
    owner = a.name.split(".")[-2] if "." in a.name else None
    w = re.escape(leaf)
    def_re = re.compile(rf"^\s*(async\s+def|def|class)\s+{w}\b|^\s*{w}\s*(:[^=]*)?=(?!=)")
    use_re = re.compile(rf"\b{w}\b")
    defs, uses = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root)
            test = is_test(rel)
            try:
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            current_class = None
            for i, line in enumerate(lines, 1):
                m = re.match(r"^class\s+(\w+)", line)
                if m:
                    current_class = m.group(1)
                if not use_re.search(line):
                    continue
                if def_re.search(line) and not test:
                    ctx = f" (in class {current_class})" if current_class and line.startswith((" ", "\t")) else ""
                    if owner is None or owner == current_class or not ctx:
                        defs.append(f"{rel}:{i}: {line.strip()[:160]}{ctx}")
                        continue
                if test and not a.include_tests:
                    continue
                uses.append(f"{rel}:{i}: {line.strip()[:160]}")
    print(f"DEFINITIONS of {a.name} ({len(defs)}):")
    for d in defs[:20]:
        print("  " + d)
    print(f"USAGES ({len(uses)}{'' if a.include_tests else ', tests excluded'}):")
    for u in uses[: a.max_usages]:
        print("  " + u)
    if len(uses) > a.max_usages:
        print(f"  ... {len(uses) - a.max_usages} more")


if __name__ == "__main__":
    sys.exit(main())
