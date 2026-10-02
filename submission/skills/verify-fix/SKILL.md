---
name: verify-fix
description: Check a fix before finishing. check_patch.py inspects the pending git diff (syntax errors, scratch or __pycache__ files, edited tests, empty patch); run_tests.py runs targeted pytest with a timeout and a short summary.
---

# verify-fix

Pass `args` as a list of strings. Both scripts operate on `/workspace`.

| Script | Purpose | args example |
|---|---|---|
| `scripts/check_patch.py` | Lint the pending `git diff HEAD`. Fix every ERROR before finishing. | `[]` |
| `scripts/run_tests.py` | `pytest -x -q` on given targets with a timeout; prints summary + first failure. | `["--targets", "tests/test_utils.py", "--keyword", "merge"]` |
