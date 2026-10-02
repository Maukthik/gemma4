#!/usr/bin/env python3
"""Validate the submission directory and package it as submission.zip.

Checks performed:
  * agent.yaml exists at the root and parses (with `!include` resolved).
  * every `!include` resolves inside the submission root (relative to the
    including file; `..` is fine if it stays inside), and every `config_path` /
    skill path is relative with no `..` component at all (adk_submission rule).
  * `tools` entries are plain tool names or `agent_tool: {config_path: ...}`.
  * prompts contain no `{placeholder}` other than the session-state keys the
    harness provides (problem_description, hints).
  * generation params are within the scorer's bounds; eval_config.yaml has only
    the four fields the scorer reads.
  * every LlmAgent uses the only allowed model.
  * every `adapter:` reference has adapters/<name>/adapter_config.json and
    adapter_model.safetensors.
  * every skill directory has SKILL.md whose frontmatter `name` equals the
    directory name.

Usage: python tools/build_submission.py [--src submission] [--out submission.zip]
"""
import argparse
import os
import pathlib
import re
import sys
import zipfile

import yaml

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
errors: list[str] = []


def inside(root: pathlib.Path, p: pathlib.Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def make_loader(root: pathlib.Path, base: pathlib.Path):
    class Loader(yaml.SafeLoader):
        pass

    def include(loader, node):
        rel = loader.construct_scalar(node)
        target = (base / rel)
        if not inside(root, target):
            errors.append(f"!include escapes submission root: {rel} (from {base})")
            return ""
        if not target.is_file():
            errors.append(f"!include target missing: {target.relative_to(root)}")
            return ""
        if target.suffix in (".yaml", ".yml"):
            return load_yaml(root, target)
        return target.read_text(encoding="utf-8")

    Loader.add_constructor("!include", include)
    return Loader


def load_yaml(root: pathlib.Path, path: pathlib.Path):
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=make_loader(root, path.parent))  # noqa: S506 (custom SafeLoader)


HARNESS_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "search_similar_code",
                 "get_code_neighbors", "get_code_subgraph", "get_status", "submit_patch"}
STATE_KEYS = {"problem_description", "hints"}
LLM_FIELDS = {"name", "description", "agent_class", "model", "adapter", "instruction", "output_key",
              "include_contents", "disallow_transfer_to_parent", "disallow_transfer_to_peers", "tools",
              "skills", "sub_agents", "generate_content_config", "before_agent_callbacks",
              "after_agent_callbacks", "before_model_callbacks", "after_model_callbacks",
              "before_tool_callbacks", "after_tool_callbacks"}
GEN_FIELDS = {"temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences", "presence_penalty",
              "frequency_penalty", "response_mime_type", "seed", "thinking_config"}


def lexical_ok(rel: str, what: str, src) -> bool:
    if rel.startswith(("/", "\\")) or ".." in pathlib.PurePosixPath(rel).parts:
        errors.append(f"{src}: {what} must be relative without '..': {rel}")
        return False
    return True


def check_instruction(text: str, src):
    # ADK substitutes {identifier} / {identifier?} from session state; unknown keys raise.
    for m in re.finditer(r"{+([^{}]*)}+", text):
        key = m.group(1).strip().rstrip("?")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and key not in STATE_KEYS and not m.group(1).strip().endswith("?"):
            errors.append(f"{src}: instruction contains {{{key}}}, which is not a session-state key "
                          f"({', '.join(sorted(STATE_KEYS))}); ADK will fail at runtime")


def check_gen(cfg, src):
    if not cfg:
        return
    if not isinstance(cfg, dict):
        errors.append(f"{src}: generate_content_config must be a mapping")
        return
    for k in cfg:
        if k not in GEN_FIELDS:
            errors.append(f"{src}: generate_content_config.{k} is not allowed")
    mot = cfg.get("max_output_tokens")
    if mot is not None and not (1 <= mot <= 32768):
        errors.append(f"{src}: max_output_tokens must be 1..32768")
    tc = cfg.get("thinking_config") or {}
    if tc.get("thinking_budget") is not None and not (1 <= tc["thinking_budget"] <= 32768):
        errors.append(f"{src}: thinking_budget must be 1..32768")


def check_agent(root: pathlib.Path, path: pathlib.Path, cfg: dict, seen: set):
    rel = path.relative_to(root)
    if path in seen:
        return
    seen.add(path)
    if not isinstance(cfg, dict):
        errors.append(f"{rel}: not a mapping")
        return
    cls = cfg.get("agent_class") or "LlmAgent"
    refs = [s["config_path"] for s in cfg.get("sub_agents") or [] if isinstance(s, dict) and "config_path" in s]
    if cls == "LlmAgent":
        for k in cfg:
            if k not in LLM_FIELDS:
                errors.append(f"{rel}: unknown LlmAgent field `{k}`")
        if not cfg.get("name") or not cfg.get("instruction"):
            errors.append(f"{rel}: `name` and `instruction` are required")
        if cfg.get("model") != MODEL:
            errors.append(f"{rel}: model must be {MODEL!r}, got {cfg.get('model')!r}")
        check_instruction(cfg.get("instruction") or "", rel)
        check_gen(cfg.get("generate_content_config"), rel)
        adapter = cfg.get("adapter")
        if adapter:
            d = root / "adapters" / adapter
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (d / f).is_file():
                    errors.append(f"{rel}: adapter {adapter!r} missing {d.relative_to(root)}/{f}")
        for tool in cfg.get("tools") or []:
            if isinstance(tool, str):
                if tool not in HARNESS_TOOLS:
                    errors.append(f"{rel}: unknown tool {tool!r} (harness tools: {sorted(HARNESS_TOOLS)})")
            elif isinstance(tool, dict) and set(tool) == {"agent_tool"} and isinstance(tool["agent_tool"], dict):
                at = tool["agent_tool"]
                extra = set(at) - {"config_path", "skip_summarization"}
                if extra or "config_path" not in at:
                    errors.append(f"{rel}: agent_tool takes config_path (+ skip_summarization), got {sorted(at)}")
                else:
                    refs.append(at["config_path"])
            else:
                errors.append(f"{rel}: tools entries must be a tool name or `agent_tool: {{config_path: ...}}`, got {tool!r}")
        for sk in cfg.get("skills") or []:
            if lexical_ok(sk, "skill path", rel) and not (root / sk / "SKILL.md").is_file():
                errors.append(f"{rel}: skill {sk!r} has no SKILL.md")
    for ref in refs:
        if not lexical_ok(ref, "config_path", rel):
            continue
        # adk_submission tries the referencing file's directory first, then the root.
        target = next((c for c in (path.parent / ref, root / ref) if c.is_file() and inside(root, c)), None)
        if target is None:
            errors.append(f"{rel}: sub-agent config not found: {ref}")
            continue
        check_agent(root, target, load_yaml(root, target), seen)


def check_eval_config(root: pathlib.Path):
    p = root / "eval_config.yaml"
    if not p.is_file():
        print("WARNING: no eval_config.yaml -> scorer uses no per-task limit; risk of the 12 h cap")
        return
    ev = (yaml.safe_load(p.read_text()) or {}).get("evaluation") or {}
    allowed = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}
    for k in ev:
        if k not in allowed:
            errors.append(f"eval_config.yaml: evaluation.{k} is ignored by the scorer")
    mins = ev.get("max_time_minutes")
    if mins is None:
        print("WARNING: eval_config.yaml has no max_time_minutes; 120 sequential tasks may exceed 12 h")
    elif mins * 120 / 60 + 120 * 0.75 / 60 > 12:
        errors.append(f"eval_config.yaml: max_time_minutes={mins} x 120 tasks + overhead exceeds the 12 h cap")
    if (ev.get("timeout_seconds") or 180) < 180:
        print("WARNING: timeout_seconds < 180 also limits hidden-test verification")


def check_skills(root: pathlib.Path):
    sk = root / "skills"
    if not sk.is_dir():
        return
    for d in sorted(p for p in sk.iterdir() if p.is_dir()):
        md = d / "SKILL.md"
        if not md.is_file():
            errors.append(f"skills/{d.name}: missing SKILL.md")
            continue
        m = re.match(r"^---\n(.*?)\n---\n", md.read_text(encoding="utf-8"), re.S)
        if not m:
            errors.append(f"skills/{d.name}/SKILL.md: missing YAML frontmatter")
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        if fm.get("name") != d.name:
            errors.append(f"skills/{d.name}: frontmatter name {fm.get('name')!r} != directory name")
        if not fm.get("description"):
            errors.append(f"skills/{d.name}: frontmatter description is required")
        elif len(fm["description"]) > 1024:
            errors.append(f"skills/{d.name}: description longer than 1024 chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="submission")
    ap.add_argument("--out", default="submission.zip")
    a = ap.parse_args()
    root = pathlib.Path(a.src).resolve()
    agent = root / "agent.yaml"
    if not agent.is_file():
        sys.exit(f"ERROR: {agent} not found")
    for p in root.rglob("*"):
        if p.is_symlink():
            errors.append(f"symlink not allowed: {p.relative_to(root)}")
    check_agent(root, agent, load_yaml(root, agent), set())
    check_skills(root)
    check_eval_config(root)
    allowed_ext = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}  # swegemma ALLOWED_SUBMISSION_EXTENSIONS
    for p in root.rglob("*"):
        if p.is_file() and p.suffix not in allowed_ext and "__pycache__" not in p.parts and p.name != ".gitkeep":
            errors.append(f"file type not allowed in submission: {p.relative_to(root)}")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if size >= 3 * 1024 ** 3:
        errors.append(f"submission is {size / 1024 ** 3:.2f} GiB; limit is 3 GiB")
    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitkeep":
                zf.write(p, p.relative_to(root).as_posix())
                n += 1
    print(f"OK: wrote {a.out} ({n} files, {os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
