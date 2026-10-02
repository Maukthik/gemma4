# Fix checklist

1. Restate the issue in one sentence: input → actual behaviour → expected behaviour.
2. List the exact identifiers in the issue (function, class, kwarg, header, CLI flag,
   error message). The hidden tests almost always use those exact names.
3. Locate the definition (`find_symbol.py`, `grep -rn`, `search_similar_code`).
4. Find every code path that needs the change: `get_code_neighbors` on the function;
   look for sync/async twins, subclasses, `__init__` re-exports, helper duplicates.
5. Reproduce with a 5–15 line script if cheap.
6. Make the minimal edit. Preserve behaviour for every input the issue doesn't mention.
7. Rerun the repro script, then the module's existing tests (`find_tests.py`).
8. `rm` scratch files, `git status`, `git diff`, `submit_patch()`.

# Feature requests
- New keyword argument: add it with a default that keeps old behaviour; thread it
  through every layer (public API → internal helper) and any `__init__`/`__all__`
  exports; update type hints/stubs if the repo has them.
- New public function/class: export it where siblings are exported.

# When stuck
- Read the traceback bottom-up and open the frame inside the library.
- `git log --oneline -5 -- <file>` shows how the file usually changes.
- Ask `code_analyzer` one precise question.
- With low budget: submit the most plausible minimal fix rather than nothing.
