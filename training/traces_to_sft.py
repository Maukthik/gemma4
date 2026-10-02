#!/usr/bin/env python3
"""Turn successful agent runs into SFT data (rejection-sampling fine-tuning).

Run the agent over the training tasks several times with the official harness:

    swegemma eval --tasks tasks.jsonl --snapshots-dir snapshots \
        --submission-dir submission --results-dir runs/r1 --sandbox docker ...

Each results dir contains `task_results.jsonl` (with `resolved`) and
`traces/trace_<instance_id>.json` (ATIF v1.7). This script keeps the traces of
resolved tasks only, converts them to the same chat format as
build_sft_data.py, and writes JSONL ready for train_lora.py.

These on-policy trajectories include real exploration and recovery from
errors, so they're better training data than the scripted ones. Mix the two
and prefer real ones when both exist.

Usage:
  python training/traces_to_sft.py --runs runs/r1 runs/r2 --out training/data/rft.jsonl \
      [--exclude training/data/holdout_ids.txt] [--max-per-task 2] [--max-calls 30]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_sft_data import TOOLS  # noqa: E402

ROOT_AGENT = "swe_coder"


def _text(msg):
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):  # list of content parts
        return "".join(p.get("text", "") for p in msg if isinstance(p, dict))
    return ""


def _obs_text(obs):
    c = obs.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c if isinstance(c, str) else json.dumps(c)


def atif_to_messages(traj, root_agent=ROOT_AGENT):
    """ATIF steps -> HF chat messages. Returns (messages, n_tool_calls)."""
    msgs, pending, n_calls = [], [], 0  # pending = tool_call ids still awaiting a response
    last_names = {}
    for step in traj.get("steps", []):
        src = step.get("source")
        extra = step.get("extra") or {}
        etype = extra.get("event_type")
        author = extra.get("author")
        if src == "agent" and author and author != root_agent:
            continue  # sub-agent internals are not part of the root agent's context
        if src == "system" and etype == "system_instruction":
            msgs.insert(0, {"role": "system", "content": _text(step.get("message"))})
            continue
        if src == "user" or etype == "continuation_nudge":
            msgs.append({"role": "user", "content": _text(step.get("message"))})
            continue
        if src == "agent" and step.get("tool_calls"):
            calls = []
            for tc in step["tool_calls"]:
                tca = (tc.get("extra") or {}).get("author")
                if tca and tca != root_agent:
                    continue
                calls.append({"id": tc["tool_call_id"], "type": "function",
                              "function": {"name": tc["function_name"], "arguments": tc.get("arguments") or {}}})
            if not calls:
                continue
            n_calls += len(calls)
            # merge with a preceding text-only assistant message (the "thought")
            if msgs and msgs[-1]["role"] == "assistant" and "tool_calls" not in msgs[-1]:
                msgs[-1]["tool_calls"] = calls
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
            pending = [c["id"] for c in calls]
            names = {c["id"]: c["function"]["name"] for c in calls}
            if step.get("observation"):
                cid = pending.pop(0)
                msgs.append({"role": "tool", "tool_call_id": cid, "name": names[cid],
                             "content": _obs_text(step["observation"])})
            last_names = names
            continue
        if src == "system" and step.get("observation") and pending:
            cid = pending.pop(0)
            msgs.append({"role": "tool", "tool_call_id": cid, "name": last_names.get(cid, ""),
                         "content": _obs_text(step["observation"])})
            continue
        if src == "agent" and etype in ("text", "final"):
            text = _text(step.get("message")).strip()
            if text:
                msgs.append({"role": "assistant", "content": text})
            continue
        # thinking, usage, compaction and other custom events are dropped
    # Drop "stopped early -> harness nudge" exchanges: we don't want to teach giving up.
    cleaned = []
    for m in msgs:
        if (m["role"] == "user" and cleaned and cleaned[-1]["role"] == "assistant"
                and "tool_calls" not in cleaned[-1] and len([x for x in cleaned if x["role"] == "user"]) >= 1):
            cleaned.pop()
            continue
        cleaned.append(m)
    # The harness records the raw instruction template; fill in the session-state key.
    if cleaned and cleaned[0]["role"] == "system" and "{problem_description}" in cleaned[0]["content"]:
        user = next((m["content"] for m in cleaned if m["role"] == "user"), "")
        ps = user.split("Problem Statement:\n", 1)[-1].split("\n## ", 1)[0].rstrip("\n")
        cleaned[0]["content"] = cleaned[0]["content"].replace("{problem_description}", ps)
    return cleaned, n_calls


def valid(msgs):
    """Every tool call answered, conversation ends with a text-only assistant turn."""
    if not msgs or msgs[0]["role"] != "system" or not any(m["role"] == "user" for m in msgs):
        return False
    open_ids = set()
    for m in msgs:
        if m["role"] == "assistant":
            if open_ids:
                return False
            open_ids = {c["id"] for c in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            open_ids.discard(m["tool_call_id"])
    return not open_ids and msgs[-1]["role"] == "assistant" and "tool_calls" not in msgs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="swegemma --results-dir folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", help="file of instance ids to leave out (holdout)")
    ap.add_argument("--max-per-task", type=int, default=2, help="cap per task so easy tasks don't dominate")
    ap.add_argument("--max-calls", type=int, default=30, help="drop very long trajectories")
    ap.add_argument("--root-agent", default=ROOT_AGENT)
    a = ap.parse_args()
    exclude = set(open(a.exclude).read().split()) if a.exclude else set()
    per_task, kept, seen = {}, 0, 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for run in a.runs:
            run = pathlib.Path(run)
            results = [json.loads(l) for l in open(run / "task_results.jsonl") if l.strip()]
            for r in results:
                iid = r["instance_id"]
                seen += 1
                if not r.get("resolved") or iid in exclude or per_task.get(iid, 0) >= a.max_per_task:
                    continue
                tp = run / "traces" / f"trace_{iid}.json"
                if not tp.exists():
                    print(f"missing trace for {iid} in {run}")
                    continue
                traj = json.load(open(tp))
                msgs, n = atif_to_messages(traj, a.root_agent)
                if n > a.max_calls or not valid(msgs):
                    print(f"skip {iid} ({run.name}): calls={n} valid={valid(msgs)}")
                    continue
                tools = (traj.get("agent") or {}).get("tool_definitions") or TOOLS
                fh.write(json.dumps({"instance_id": iid, "source": str(run), "messages": msgs, "tools": tools}) + "\n")
                per_task[iid] = per_task.get(iid, 0) + 1
                kept += 1
    print(f"kept {kept} trajectories from {seen} task results ({len(per_task)} distinct tasks) -> {a.out}")


if __name__ == "__main__":
    main()
