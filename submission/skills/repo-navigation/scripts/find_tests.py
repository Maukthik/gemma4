#!/usr/bin/env python3
"""Find test files / test functions related to a source file or symbol."""
import argparse
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="source file, e.g. requests/sessions.py")
    ap.add_argument("--name", help="symbol, e.g. Session.request")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if not a.path and not a.name:
        print("give --path and/or --name")
        return 2
    root = a.root if os.path.isdir(a.root) else "/workspace"
    needles = []
    if a.path:
        mod = os.path.splitext(a.path.replace("\\", "/"))[0]
        for prefix in ("src/",):
            if mod.startswith(prefix):
                mod = mod[len(prefix):]
        dotted = mod.replace("/", ".")
        stem = os.path.basename(mod)
        needles += [dotted, stem]
    if a.name:
        needles += [a.name.split(".")[-1]]
    pats = [re.compile(rf"\b{re.escape(n)}\b") for n in needles if n and n != "__init__"]
    stem = os.path.basename(os.path.splitext(a.path)[0]) if a.path else None
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".tox", ".venv")]
        for f in filenames:
            if not (f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py"))):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            score = sum(len(pt.findall(text)) for pt in pats)
            if stem and stem in f:
                score += 50
            if score:
                tests = []
                if a.name:
                    leaf = a.name.split(".")[-1]
                    cur = None
                    for line in text.splitlines():
                        m = re.match(r"\s*(?:async\s+)?def\s+(test\w*)", line)
                        if m:
                            cur = m.group(1)
                        elif cur and leaf in line and cur not in tests:
                            tests.append(cur)
                results.append((score, rel, tests))
    results.sort(reverse=True)
    if not results:
        print("no related tests found; try: ls tests/")
    for score, rel, tests in results[:10]:
        extra = f"  tests mentioning name: {', '.join(tests[:8])}" if tests else ""
        print(f"{rel} (score {score}){extra}")
    if results:
        print(f"\nrun: timeout 300 python -m pytest -x -q {results[0][1]} 2>&1 | tail -30")


if __name__ == "__main__":
    sys.exit(main())
