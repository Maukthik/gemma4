#!/usr/bin/env python3
"""Run our submission through the official harness on a Kaggle L4x4 notebook.

Based on the organizers' "getting-started-gemma-4-developer-agent" notebook:
install the wheelhouse, serve the QAT model with vLLM (same settings as the
scorer), then run swegemma's Evaluator (Phase 1 agent + Phase 2 hidden tests).

Two uses:
  1. Measure:  --task-ids training/data/holdout_ids.txt   (score on held-out tasks)
  2. Sample for rejection-sampling SFT: --repeat 4 --temperature 0.7 on the train
     tasks, then `python training/traces_to_sft.py --runs results/run_*`.

Notebook settings: accelerator "GPU L4 x4", internet OFF; attach the competition
data, the model google/gemma-4 (gemma-4-31b-it-qat-w4a16-ct) and the dataset
metric/gemma-4-developer-agent-wheelhouse. Check the input paths below, since
Kaggle mounts can differ between notebooks.

Usage (in a notebook cell):
  !python kaggle/run_agent_eval.py --submission submission --task-ids training/data/holdout_ids.txt
"""
import argparse
import asyncio
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import yaml

WHEELHOUSE = pathlib.Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
DATA_DIR = pathlib.Path("/kaggle/input/competitions/gemma-4-developer-agent")
MODEL_PATH = pathlib.Path("/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2")
MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"


def install_wheelhouse():
    for k, v in {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1",
                 "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
                 "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_NO_USAGE_STATS": "1", "OTEL_SDK_DISABLED": "true",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
        os.environ[k] = v
    try:
        import swegemma  # noqa: F401
        return
    except ImportError:
        pass
    for pth in glob.glob("/usr/local/lib/python*/*-packages/*cutlass*.pth"):
        try:
            os.unlink(pth)
        except OSError:
            pass
    tmp = pathlib.Path("/tmp/wheelhouse")
    tmp.mkdir(parents=True, exist_ok=True)
    for w in WHEELHOUSE.glob("*.whl"):
        if "cutlass" in w.name.lower():
            continue
        name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
        if not (tmp / name).exists():
            os.symlink(w, tmp / name)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                    *sorted(map(str, tmp.glob("*.whl")))], check=True)


def prepare_submission(src, temperature, seed):
    """Copy the submission; optionally override sampling for diverse rollouts."""
    if temperature is None and seed is None:
        return pathlib.Path(src)
    dst = pathlib.Path(tempfile.mkdtemp(prefix="sub_")) / "submission"
    shutil.copytree(src, dst)
    sp = dst / "configs" / "sampling.yaml"
    cfg = yaml.safe_load(sp.read_text())
    if temperature is not None:
        cfg["temperature"] = temperature
    if seed is not None:
        cfg["seed"] = seed
    sp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--task-ids", help="file with instance ids (default: all tasks)")
    ap.add_argument("--exclude-ids", help="file with instance ids to skip (e.g. holdout when sampling)")
    ap.add_argument("--results-dir", default="/kaggle/working/results")
    ap.add_argument("--repeat", type=int, default=1, help="independent runs (results_dir/run_<i>)")
    ap.add_argument("--temperature", type=float, help="override sampling temperature (rollouts)")
    ap.add_argument("--concurrency", type=int, default=1, help="scorer runs sequentially; >1 only for sampling")
    a = ap.parse_args()

    install_wheelhouse()
    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks
    from swegemma.models.discovery import validate_single_declared_model

    litellm.drop_params = True
    tasks = load_tasks(DATA_DIR / "tasks.jsonl")
    ids = [t.instance_id for t in tasks]
    if a.task_ids:
        keep = set(open(a.task_ids).read().split())
        ids = [i for i in ids if i in keep]
    if a.exclude_ids:
        drop = set(open(a.exclude_ids).read().split())
        ids = [i for i in ids if i not in drop]
    print(f"{len(ids)} tasks x {a.repeat} run(s)")

    sub0 = pathlib.Path(a.submission).resolve()
    declared = validate_single_declared_model(sub0)
    adapters = discover_adapters(str(sub0), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    gpus = torch.cuda.device_count()
    server = VllmServer(VllmConfig(
        model=str(MODEL_PATH), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
        default_chat_template_kwargs={"enable_thinking": True}, max_model_len=32768, dtype="bfloat16",
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True, max_loras=8,
        max_lora_rank=128, tensor_parallel_size=4 if gpus >= 4 else max(1, gpus), startup_timeout=60 * 20,
    ), adapter_manifest=adapters)
    server.start()
    try:
        models = server.create_model_registry(aliases=[declared, MODEL_NAME], model_prefix="openai/", api_key="EMPTY")
        ev = (yaml.safe_load((sub0 / "eval_config.yaml").read_text()) or {}).get("evaluation", {})
        limits, gen_constraints = build_submission_limits()
        for r in range(a.repeat):
            sub = prepare_submission(sub0, a.temperature, (1000 + r) if a.repeat > 1 else None)
            out = pathlib.Path(a.results_dir) / (f"run_{r}" if a.repeat > 1 else "")
            cfg = EvalConfig(
                tasks_path=DATA_DIR / "tasks.jsonl", snapshots_dir=DATA_DIR / "snapshots", results_dir=out,
                submission_dir=sub, models=models, sandbox="subprocess", task_ids=ids,
                timeout_seconds=int(ev.get("timeout_seconds", 180)),
                max_time_minutes=float(ev.get("max_time_minutes", 4.5)),
                max_tool_calls=ev.get("max_tool_calls"), max_turns=ev.get("max_turns"),
                limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                                token_threshold=32768, event_retention_size=5),
                graph_dir=str(DATA_DIR / "graphs"), embeddings_dir=str(DATA_DIR / "embeddings"),
                wheels_dir=DATA_DIR / "wheels", concurrency=a.concurrency, display_mode="quiet",
            )
            res = asyncio.run(Evaluator(cfg).run())
            n = len(res.task_results)
            solved = sum(bool(t.resolved) for t in res.task_results)
            print(f"run {r}: resolved {solved}/{n} = {100 * solved / max(1, n):.1f}%  -> {out}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
