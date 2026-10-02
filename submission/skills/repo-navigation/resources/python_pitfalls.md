# Pitfalls that make hidden tests fail

- Fixing only one of the sync / async implementations (httpx `Client` vs `AsyncClient`,
  starlette/fastapi sync vs async endpoints, `iter_*` vs `aiter_*`).
- Changing an exception type or message that existing tests match with `pytest.raises(..., match=...)`.
- Mutating a shared default (`def f(x=[])`, class-level dicts) instead of copying.
- Treating falsy values as missing: use `is None`, not `if not value`, when `0`, `""`, `[]`
  or `False` are valid inputs.
- `bytes` vs `str`: headers, URLs and bodies often accept both; normalise consistently.
- Case-insensitive header handling; preserve the original casing when echoing back.
- Breaking `repr()`/`str()` output that tests compare literally.
- Forgetting `__all__`/top-level re-exports for a new public name.
- Pydantic v1 vs v2 code paths in FastAPI (`PYDANTIC_V2` branches): update both.
- Rich renderables: width/measurement must stay consistent with `__rich_console__`
  output; check `Segment` styles and `no_wrap`/`overflow` handling.
- Leaving debug prints, repro scripts or new untracked files in /workspace (write scratch to /tmp)
  (`submit_patch` stages untracked files too).
