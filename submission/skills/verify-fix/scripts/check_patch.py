#!/usr/bin/env python3
"""Sanity-check the pending diff before submit_patch."""
import os
import re
import subprocess
import sys

SCRATCH = re.compile(r"\.py[co]$|(^|/)__pycache__/|(^|/)(repro|reproduce|scratch|debug|tmp|test_issue|issue)[\w-]*\.(py|txt|log)$|\.orig$|\.rej$|\.bak$")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def main():
    cwd = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
    status = git("status", "--porcelain", "--untracked-files=all", cwd=cwd).splitlines()
    files = [l[3:].strip() for l in status if l.strip()]
    errors, warnings = [], []
    if not files:
        errors.append("no changes at all: the patch would be empty")
    src_changed = False
    for f in files:
        path = os.path.join(cwd, f)
        is_test = bool(re.search(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$", f))
        if SCRATCH.search(f):
            errors.append(f"scratch file will be submitted: {f} (rm it)")
        if f.endswith(".py") and os.path.exists(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    compile(fh.read(), f, "exec", dont_inherit=True)  # in memory: no __pycache__
            except SyntaxError as e:
                errors.append(f"syntax error in {f}:{e.lineno}: {e.msg}")
            if not is_test:
                src_changed = True
    diff = git("diff", "HEAD", cwd=cwd)
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if re.search(r"\b(breakpoint\(\)|pdb\.set_trace|print\(\s*['\"]DEBUG)", line):
                warnings.append(f"debug statement added: {line[1:].strip()[:120]}")
    if files and not src_changed:
        errors.append("no library source file changed (only tests/other files)")
    print("CHANGED FILES:", ", ".join(files) if files else "(none)")
    print(f"DIFF LINES: +{sum(1 for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++'))}"
          f" -{sum(1 for l in diff.splitlines() if l.startswith('-') and not l.startswith('---'))}")
    for e in errors:
        print("ERROR:", e)
    for w in warnings:
        print("WARNING:", w)
    print("OK: ready to submit_patch" if not errors else "FIX THE ERRORS ABOVE BEFORE submit_patch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
