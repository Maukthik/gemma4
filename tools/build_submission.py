#!/usr/bin/env python3
"""Validate the submission directory and package it as submission.zip.

Checks performed:
  * agent.yaml exists at the root and parses (with `!include` resolved).
  * every `!include` / `config_path` resolves to a file inside the submission
    root (no `..` escapes, no symlinks).
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


def check_agent(root: pathlib.Path, path: pathlib.Path, cfg: dict, seen: set):
    rel = path.relative_to(root)
    if path in seen:
        return
    seen.add(path)
    if not isinstance(cfg, dict):
        errors.append(f"{rel}: not a mapping")
        return
    for key in ("name", "instruction"):
        if not cfg.get(key):
            errors.append(f"{rel}: missing `{key}`")
    if cfg.get("agent_class", "LlmAgent") == "LlmAgent" and cfg.get("model") != MODEL:
        errors.append(f"{rel}: model must be {MODEL!r}, got {cfg.get('model')!r}")
    adapter = cfg.get("adapter")
    if adapter:
        d = root / "adapters" / adapter
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            if not (d / f).is_file():
                errors.append(f"{rel}: adapter {adapter!r} missing {d.relative_to(root)}/{f}")
    refs = []
    for sub in cfg.get("sub_agents") or []:
        if isinstance(sub, dict) and sub.get("config_path"):
            refs.append(sub["config_path"])
    for tool in cfg.get("tools") or []:
        if isinstance(tool, dict) and tool.get("name") == "AgentTool":
            agent = ((tool.get("args") or {}).get("agent") or {})
            if agent.get("config_path"):
                refs.append(agent["config_path"])
    for ref in refs:
        # ADK resolves config_path relative to the referencing file; also accept root-relative.
        cands = [path.parent / ref, root / ref]
        target = next((c for c in cands if c.is_file() and inside(root, c)), None)
        if target is None:
            errors.append(f"{rel}: sub-agent config not found inside root: {ref}")
            continue
        check_agent(root, target, load_yaml(root, target), seen)


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
