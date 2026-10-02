#!/usr/bin/env python3
"""Run pytest with a timeout and print a compact summary."""
import argparse
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="", help="space-separated test paths / node ids")
    ap.add_argument("--keyword", default="", help="pytest -k expression")
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--tail", type=int, default=15)
    a = ap.parse_args()
    cwd = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", "-rf"]
    cmd += a.targets.split()
    if a.keyword:
        cmd += ["-k", a.keyword]
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=a.timeout)
        out = r.stdout + r.stderr
        code = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        print(f"TIMEOUT after {a.timeout}s; narrow --targets or --keyword")
        code = -1
    lines = out.splitlines()
    summary = [l for l in lines if re.search(r"\b(passed|failed|error|no tests ran)\b", l)]
    print("COMMAND:", " ".join(cmd[1:]))
    print("EXIT:", code, "(0=all passed, 1=failures, 2=interrupted/collection error, 5=no tests)")
    if summary:
        print("SUMMARY:", summary[-1].strip())
    if code != 0:
        # show the first failure block + tail
        start = next((i for i, l in enumerate(lines) if l.startswith(("____", "E ", "ERROR"))), None)
        if start is not None:
            print("\n".join(lines[start:start + 25]))
            print("...")
    print("\n".join(lines[-a.tail:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
