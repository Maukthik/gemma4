---
name: repo-navigation
description: Fast, low-token orientation in an unfamiliar Python repository. Gives a compact package map, finds where a symbol is defined and used, and finds the test files that cover a source file. Use at the start of a task or when grep output is too noisy.
---

# repo-navigation

All scripts are read-only and print compact, truncated output. Run them with
`run_skill_script`, passing options as long arguments.

| Script | Purpose | Example args |
|---|---|---|
| `scripts/repo_map.py` | Package layout: each module with its top-level classes/functions (one line each). | `--root . --max-lines 120` or `--root src/httpx` |
| `scripts/find_symbol.py` | Definitions (`def`/`class`/assignment) and usages of a name, tests excluded by default. | `--name merge_setting` `--include-tests` |
| `scripts/find_tests.py` | Test files and test functions that reference a source module or symbol. | `--path requests/sessions.py` or `--name Session.request` |

Recommended order for a new issue:
1. `find_symbol.py --name <identifier from the issue>`
2. `read_file` the definition it reports.
3. `find_tests.py --path <that file>` to know which tests to run after the fix.

Resources: `resources/workflow.md` (fix checklist), `resources/python_pitfalls.md`
(common mistakes that make hidden tests fail). Load them with
`load_skill_resource` if unsure what to do next.
