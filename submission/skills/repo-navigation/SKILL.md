---
name: repo-navigation
description: Low-token orientation in an unfamiliar Python repository. Finds where a symbol is defined and used (tests excluded), finds the test files covering a source file, and prints a compact package map. Use when grep output is too noisy.
---

# repo-navigation

Read-only scripts with compact, truncated output. They always operate on
`/workspace`. Pass `args` as a list of strings.

| Script | Purpose | args example |
|---|---|---|
| `scripts/find_symbol.py` | Definitions (`def`/`class`/assignment) and non-test usages of a name. `Class.method` narrows to that class. | `["--name", "merge_setting"]`, `["--name", "Client.send", "--include-tests"]` |
| `scripts/find_tests.py` | Test files (and test functions) that reference a source module or symbol, plus a ready pytest command. | `["--path", "requests/sessions.py"]`, `["--name", "Session.request"]` |
| `scripts/repo_map.py` | Module list with top-level classes/functions, one line each. | `["--root", "src/httpx"]` |

Typical use: `find_symbol.py` on the identifier from the issue, then `read_file`
the reported lines, then `find_tests.py` on that file to know what to run.

Resources (`load_skill_resource`): `resources/workflow.md` (fix checklist),
`resources/python_pitfalls.md` (mistakes that make hidden tests fail).
