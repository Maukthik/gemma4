---
name: verify-fix
description: Verify a fix before submitting. Runs the relevant pytest targets with a timeout and compact output, and sanity-checks the pending git diff (syntax errors, leftover scratch files, edited tests, debug prints). Use after editing and right before submit_patch.
---

# verify-fix

| Script | Purpose | Example args |
|---|---|---|
| `scripts/run_tests.py` | Run pytest on given targets with a timeout; prints a short summary and the first failure. | `--targets tests/test_utils.py` `--keyword merge` `--timeout 300` |
| `scripts/check_patch.py` | Lint the pending `git diff HEAD`: changed files compile, no scratch/debug files, warns if only tests changed or no source changed. | (no args) |

Call `check_patch.py` immediately before `submit_patch`. Fix every ERROR it reports.
