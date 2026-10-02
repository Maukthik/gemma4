#!/usr/bin/env python3
"""Build supervised fine-tuning trajectories from the competition training set.

For every task in tasks.jsonl we unpack the repo snapshot and *execute* a
scripted expert policy against it, so every tool observation in the training
data is a real output of the real repository:

    think -> grep for identifiers from the issue -> read_file around each hunk
          -> edit_file (derived from the gold patch hunks) -> git diff
          -> submit_patch

The edit_file calls are checked: after replaying them, the working tree must
be byte-identical to `git apply <gold patch>`. Tasks that fail this check are
skipped.

Output: JSONL, one conversation per line, in the HF chat format
({"messages": [...], "tools": [...]}) with OpenAI-style `tool_calls`, ready
for `tokenizer.apply_chat_template(messages, tools=tools)`.

Usage:
  python training/build_sft_data.py \
      --tasks /kaggle/input/<dataset>/tasks.jsonl \
      --snapshots /kaggle/input/<dataset>/snapshots \
      --out training/data/sft.jsonl [--holdout 20]
"""
import argparse
import difflib
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SYSTEM_TEMPLATE = (HERE.parent / "submission" / "prompts" / "system.md").read_text(encoding="utf-8")
BUDGET = {"minutes": 4.5, "calls": 40, "turns": 80, "timeout": 180}  # keep in sync with eval_config.yaml

# Tool schemas mirroring the competition harness signatures.
TOOLS = [
    {"type": "function", "function": {"name": "run_command", "description": "Executes a shell command in /bin/bash -c inside /workspace.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Reads a file from /workspace with 1-indexed inclusive line slicing.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replaces old_string with new_string in an existing non-empty file inside /workspace.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "allow_multiple": {"type": "boolean"}}, "required": ["filepath", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Creates or overwrites a file at /workspace/<filepath>.",
     "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filepath", "content"]}}},
    {"type": "function", "function": {"name": "search_similar_code", "description": "Finds top-k graph nodes with highest cosine similarity to query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_code_neighbors", "description": "Finds incoming and outgoing neighbors of a symbol in the repository call/dependency graph.",
     "parameters": {"type": "object", "properties": {"node": {"type": "string"}, "edge_type": {"type": "string"}, "max_neighbors": {"type": "integer"}}, "required": ["node"]}}},
    {"type": "function", "function": {"name": "get_code_subgraph", "description": "Extracts the induced subgraph for a list of symbols.",
     "parameters": {"type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "string"}}}, "required": ["nodes"]}}},
    {"type": "function", "function": {"name": "get_status", "description": "Returns live budget consumption and patch status.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "submit_patch", "description": "Stages untracked files and captures git diff HEAD from /workspace.",
     "parameters": {"type": "object", "properties": {}}}},
    # SkillToolset tools (ADK) added because agent.yaml declares skills
    {"type": "function", "function": {"name": "list_skills", "description": "Lists all available skills with their names and descriptions.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "load_skill", "description": "Loads the SKILL.md instructions for a given skill.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "load_skill_resource", "description": "Loads a resource file (from references/, assets/, or scripts/) from within a skill.",
     "parameters": {"type": "object", "properties": {"skill_name": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["skill_name", "file_path"]}}},
    {"type": "function", "function": {"name": "run_skill_script", "description": "Executes a script from a skill's scripts/ directory.",
     "parameters": {"type": "object", "properties": {"skill_name": {"type": "string"}, "file_path": {"type": "string"},
                    "args": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "string"}}]}}, "required": ["skill_name", "file_path"]}}},
]



# --------------------------------------------------------------------------- patch parsing
def parse_patch(patch: str):
    """Return {path: [hunk, ...]} where hunk = dict(old_start, old_lines[], new_lines[])."""
    files, cur, hunk = {}, None, None
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git"):
            cur, hunk = None, None
        elif line.startswith("+++ "):
            p = line[4:].strip()
            cur = None if p == "/dev/null" else re.sub(r"^b/", "", p)
            if cur:
                files.setdefault(cur, [])
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            hunk = {"old_start": int(m.group(1)), "old": [], "new": []}
            if cur:
                files[cur].append(hunk)
        elif hunk is not None and cur is not None:
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            tag, body = line[:1], line[1:]
            if tag in (" ", ""):
                hunk["old"].append(body if tag else "\n")
                hunk["new"].append(body if tag else "\n")
            elif tag == "-":
                hunk["old"].append(body)
            elif tag == "+":
                hunk["new"].append(body)
    return files


def trim_context(old, new, keep=3):
    """Drop shared leading/trailing lines beyond `keep` so edits stay compact."""
    lead = 0
    while lead < min(len(old), len(new)) and old[lead] == new[lead]:
        lead += 1
    trail = 0
    while trail < min(len(old), len(new)) - lead and old[-1 - trail] == new[-1 - trail]:
        trail += 1
    s = max(0, lead - keep)
    e = max(0, trail - keep)
    return old[s:len(old) - e], new[s:len(new) - e], s


def make_edits(path_text: str, hunks):
    """Turn hunks into (old_string, new_string) pairs that are unique in the evolving file."""
    edits, text = [], path_text
    for h in hunks:
        old, new, _ = trim_context(h["old"], h["new"])
        old_s, new_s = "".join(old), "".join(new)
        if not old_s.strip():  # pure insertion with no context: use the full hunk
            old_s, new_s = "".join(h["old"]), "".join(h["new"])
        lines = text.splitlines(keepends=True)
        # grow context until unique
        grow = 0
        while text.count(old_s) != 1 and grow < 15:
            grow += 1
            idx = text.find(old_s)
            if idx < 0:
                return None
            start_line = text[:idx].count("\n")
            end_line = start_line + old_s.count("\n")
            pre = "".join(lines[max(0, start_line - 1):start_line])
            post = "".join(lines[end_line:end_line + 1])
            old_s, new_s = pre + old_s + post, pre + new_s + post
        if text.count(old_s) != 1:
            return None
        edits.append((old_s, new_s))
        text = text.replace(old_s, new_s, 1)
    return edits, text


# --------------------------------------------------------------------------- harness emulation
# Observations mirror swegemma's JSON tool responses (ok_response / error_response)
# and its limits: 5000-char command output, 150-line / 10k-char read_file.
MAX_STDOUT = 5000


def ok(**kw):
    return json.dumps({"status": "ok", **kw})


def err(error_type, message, details=None):
    d = {"status": "error", "error_type": error_type, "error_message": message}
    if details is not None:
        d["details"] = details
    return json.dumps(d)


def sh(cmd, cwd, timeout=60):
    """run_command semantics: non-zero exit -> CommandError with stdout/stderr details."""
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except subprocess.TimeoutExpired:
        return err("TimeoutExceeded", f"Command timed out after {timeout} seconds")
    out, errs = r.stdout[:MAX_STDOUT], r.stderr[:MAX_STDOUT]
    if r.returncode != 0:
        return err("CommandError", errs or out, {"stdout": out, "stderr": errs, "exit_code": r.returncode})
    return ok(stdout=out, stderr=errs, exit_code=0)


def raw(cmd, cwd):
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True).stdout


def read_obs(root, path, start, end):
    lines = (root / path).read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start, end = max(1, start), min(total, end)
    trunc = False
    if end - start + 1 > 150:
        end, trunc = start + 149, True
    snippet = "\n".join(lines[start - 1:end])
    if len(snippet) > 10000:
        snippet, trunc = snippet[:10000], True
    return ok(filepath=path, content=snippet, start_line=start, end_line=end, total_lines=total, is_truncated=trunc)


def edit_obs(path, before, after):
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=f"a/{path}", tofile=f"b/{path}", n=2))
    return ok(filepath=path, occurrences=1, strategy="exact", diff=diff[:MAX_STDOUT],
              is_truncated=len(diff) > MAX_STDOUT)


def workspace_tree(root):
    out = raw("find . -maxdepth 3 -not -path './.git*' -not -name '__pycache__' | sort | head -150", root)
    return out.strip()


def format_task(task, tree, budget):
    """Same layout as swegemma.harness.agent_runner.build_agent_prompt."""
    parts = [f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
             f"Problem Statement:\n{task['problem_statement']}\n"]
    if (task.get("hints_text") or "").strip():
        parts.append(f"## Hints:\n{task['hints_text'].strip()}\n")
    parts.append("## Task Budget (Session terminates when any budget is exhausted)\n"
                 f"- Time allowance: {budget['minutes']} minutes\n- Tool calls allowance: {budget['calls']} calls\n"
                 f"- Max loop iterations: {budget['turns']} turns\n")
    parts.append("## Execution Environment Rules\n"
                 f"- Single command timeout: {budget['timeout']} seconds (commands exceeding this fail without ending the session)\n"
                 "- Command output limit: 5000 characters\n- File view limit: 150 lines per read_file call\n"
                 "- File character limit: 10000 characters per read_file call\n"
                 "- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY "
                 "pre-installed. Do NOT attempt to run pip install or download packages.\n")
    parts.append("\n".join([
        "## Instructions:",
        "0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system "
        "site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.",
        "1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI "
        "subcommands, or Python modules.",
        "2. Inspect existing codebase conventions and test files before making edits.",
        "3. Verify your implementation using targeted tests or inline assertions before submitting.",
        "4. Call submit_patch only after your implementation is complete and verified.",
        "5. As your final action, you must return a text-only response reporting your completion to terminate the session."]))
    parts.append("## Code Intelligence Tools\nThis repository has pre-built code graph and embedding data. Use these "
                 "tools for fast, targeted navigation:\n- `search_similar_code(query)`: Find semantically similar "
                 "functions/classes by keyword.\n- `get_code_neighbors(node)`: Find callers, callees, and definitions "
                 "related to a symbol.\n- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols.\n")
    parts.append("## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up to 3 "
                 "levels):\n```\n" + tree + "\n```\n")
    return "\n".join(parts)


def unpack(snapshots, iid, dest):
    tgz = pathlib.Path(snapshots) / f"{iid}.tgz"
    with tarfile.open(tgz) as tf:
        tf.extractall(dest, filter="data") if sys.version_info >= (3, 12) else tf.extractall(dest)
    # archive may contain a top-level directory
    entries = [p for p in pathlib.Path(dest).iterdir()]
    if len(entries) == 1 and entries[0].is_dir() and not (pathlib.Path(dest) / ".git").exists():
        return entries[0]
    return pathlib.Path(dest)


def is_test_path(p):
    return bool(re.search(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$", p))


def issue_identifiers(text, repo_root, files):
    """Identifiers mentioned in the issue that also occur in the files touched by the fix."""
    cands = re.findall(r"`([A-Za-z_][\w.]{2,60})`", text) + re.findall(r"\b([A-Za-z_]+_[\w]+|[a-z]+[A-Z]\w+|[A-Z][a-z]+[A-Z]\w+)\b", text)
    seen, out = set(), []
    blob = "".join((repo_root / f).read_text(encoding="utf-8", errors="replace") for f in files if (repo_root / f).exists())
    for c in cands:
        leaf = c.split(".")[-1].strip("()")
        if leaf in seen or len(leaf) < 4:
            continue
        seen.add(leaf)
        if re.search(rf"\b{re.escape(leaf)}\b", blob):
            out.append(leaf)
    return out[:3]


# --------------------------------------------------------------------------- trajectory
class Traj:
    def __init__(self, task, tree):
        system = SYSTEM_TEMPLATE.replace("{problem_description}", task["problem_statement"])
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": format_task(task, tree, BUDGET)},
        ]
        self.n = 0

    def call(self, thought, name, args, observation):
        self.n += 1
        cid = f"call_{self.n}"
        self.messages.append({"role": "assistant", "content": thought,
                              "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]})
        self.messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": observation})


def build_one(task, snapshots, rng):
    files = {p: h for p, h in parse_patch(task["patch"]).items() if not is_test_path(p)}
    if not files:
        return None, "patch touches only tests"
    tmp = tempfile.mkdtemp(prefix="sft_")
    try:
        root = unpack(snapshots, task["instance_id"], tmp)
        raw("git config user.email a@b.c && git config user.name sft", root)
        t = Traj(task, workspace_tree(root))

        existing = [p for p in files if (root / p).exists()]
        idents = issue_identifiers(task["problem_statement"], root, existing)

        # 1. localise
        if idents:
            ident = idents[0]
            cmd = f'grep -rn "{ident}" --include="*.py" . | grep -v tests/ | head -30'
            t.call(f"The issue is about `{ident}`. Let me find where it is defined and used in the library code.",
                   "run_command", {"command": cmd}, sh(cmd, root))
        else:
            cmd = "git ls-files '*.py' | grep -v tests/ | head -40"
            t.call("Let me list the library's Python source files to find the relevant module.",
                   "run_command", {"command": cmd}, sh(cmd, root))

        # 2. read every region we will touch, in windows of <= 80 lines
        for path, hunks in files.items():
            if not (root / path).exists():
                continue
            windows = []
            for h in hunks:
                a = max(1, h["old_start"] - 15)
                b = h["old_start"] + len(h["old"]) + 15
                if windows and a <= windows[-1][1] + 5 and b - windows[-1][0] <= 80:
                    windows[-1][1] = max(windows[-1][1], b)
                else:
                    windows.append([a, min(b, a + 79)])
            for i, (a, b) in enumerate(windows):
                thought = (f"`{path}` looks responsible. Let me read the relevant code." if i == 0
                           else f"Reading the other part of `{path}` that needs to change.")
                t.call(thought, "read_file", {"filepath": path, "start_line": a, "end_line": b},
                       read_obs(root, path, a, b))

        # 3. edits
        final_texts = {}
        for path, hunks in files.items():
            fp = root / path
            if not fp.exists():  # new file
                content = "".join(l for h in hunks for l in h["new"])
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                t.call(f"The fix needs a new module `{path}`.", "write_file",
                       {"filepath": path, "content": content}, ok(filepath=path, size=len(content)))
                final_texts[path] = content
                continue
            text = fp.read_text(encoding="utf-8")
            res = make_edits(text, hunks)
            if res is None:
                return None, f"could not derive unique edits for {path}"
            edits, final = res
            for i, (old, new) in enumerate(edits):
                thought = ("I understand the bug now. Making the minimal fix" if i == 0 and len(final_texts) == 0
                           else "Applying the next part of the change") + f" in `{path}`."
                cur = fp.read_text(encoding="utf-8")
                after = cur.replace(old, new, 1)
                fp.write_text(after, encoding="utf-8")
                t.call(thought, "edit_file", {"filepath": path, "old_string": old, "new_string": new},
                       edit_obs(path, cur, after))
            final_texts[path] = final

        # verify replay == gold
        mine = {p: (root / p).read_text(encoding="utf-8") for p in final_texts}
        raw("git stash -u -q", root)
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as pfh:
            pfh.write(task["patch"])
        r = subprocess.run(["git", "apply", pfh.name], cwd=root, capture_output=True, text=True)
        os.unlink(pfh.name)
        if r.returncode != 0:
            return None, "gold patch does not apply"
        for p, txt in mine.items():
            if (root / p).read_text(encoding="utf-8") != txt:
                return None, f"replayed edits differ from gold for {p}"
        for p in list(parse_patch(task["patch"])):  # agent state = gold source changes only
            if is_test_path(p):
                raw(f"git checkout -q -- '{p}' 2>/dev/null || rm -f '{p}'", root)

        # 4. submit early (free), then verify, then finish with text
        diff = raw("git add -N . && git diff HEAD", root)
        t.call("The fix is in place. Submitting now so the work is safe, then I will verify it.", "submit_patch", {},
               ok(patch_size=len(diff), files_changed=diff.count("diff --git")))
        changed = " ".join(f"'{p}'" for p in files)
        cmd = (f"python3 -c 'import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]; print(\"syntax ok\")' "
               f"{changed} && git status --short")
        t.call("Checking that the changed files parse and that only intended files are modified.", "run_command",
               {"command": cmd}, sh(cmd, root))
        summary = ", ".join(f"`{p}`" for p in files)
        t.messages.append({"role": "assistant", "content": f"Fixed the issue by updating {summary}; the patch is submitted."})
        return {"instance_id": task["instance_id"], "messages": t.messages, "tools": TOOLS}, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", default=str(HERE / "data" / "sft.jsonl"))
    ap.add_argument("--holdout", type=int, default=20, help="tasks kept out of training for local eval")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    rng.shuffle(tasks)
    holdout, train = tasks[: a.holdout], tasks[a.holdout:]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "holdout_ids.txt").write_text("\n".join(t["instance_id"] for t in holdout) + "\n")
    ok = 0
    with open(out, "w", encoding="utf-8") as fh:
        for task in train:
            try:
                ex, why = build_one(task, a.snapshots, rng)
            except Exception as e:  # keep going on odd snapshots
                ex, why = None, f"{type(e).__name__}: {e}"
            if ex is None:
                print(f"skip {task['instance_id']}: {why}")
                continue
            fh.write(json.dumps(ex) + "\n")
            ok += 1
    print(f"wrote {ok}/{len(train)} trajectories to {out}; {len(holdout)} holdout ids in {out.parent / 'holdout_ids.txt'}")


if __name__ == "__main__":
    main()
