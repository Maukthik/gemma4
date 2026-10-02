# Gemma 4 Developer Agent Competition: submission workspace

Goal: a `submission.zip` holding an ADK agent config that drives
`gemma-4-31b-it-qat-w4a16-ct`, optionally with LoRA adapters. It must fix
~120 hidden Python issues (SWE-bench-style, from private repos) offline within
**12 hours total**. Score = % of tasks whose hidden tests pass after the
patch is applied.

## What's in this repo

```
submission/                    -> zipped into submission.zip
  agent.yaml                   root coder agent (model, prompt, tools, sub-agent)
  configs/sampling.yaml        near-greedy decoding (exact strings matter for edit_file)
  prompts/system.md            the workflow: understand -> locate -> repro -> fix -> verify -> submit
  prompts/analyzer.md          read-only localisation sub-agent
  sub_agents/code_analyzer.yaml
  skills/repo-navigation/      repo_map.py, find_symbol.py, find_tests.py + checklists
  skills/verify-fix/           run_tests.py, check_patch.py (catches empty patch, scratch files,
                               __pycache__, syntax errors, test-only diffs)
  adapters/                    put trained PEFT adapters here (main_lora/)
tools/build_submission.py      validate (includes, sandboxing, model name, adapters, skills) + zip
tools/local_eval.py            local grader: apply prediction + test_patch, run pytest in sandbox image
training/build_sft_data.py     build SFT trajectories by replaying gold fixes on real snapshots
training/train_lora.py         QLoRA SFT on Kaggle L4x4, loss on assistant turns only
```

## Strategy

1. **Baseline first (no training).** The agent workflow in `prompts/system.md` plus the
   skills. Most of the score comes from the agent reliably reaching a minimal,
   correct edit and **always calling `submit_patch`**, and from not wasting the time budget.
2. **Time budget.** 12 h / ~120 tasks ≈ **6 min per task** on 4×L4. The prompt aims
   for 10–25 tool calls; set per-task limits in `eval_config.yaml` once the harness
   format is known so that one stuck task can't eat everyone else's time.
3. **SFT LoRA (`main_lora`).** Train on trajectories whose observations are real
   (`build_sft_data.py`). This mainly teaches tool-call format, exact `edit_file`
   strings and the submit habit. Hold out 20 tasks to measure.
4. **Rejection sampling / RFT (next).** Run the baseline agent several times per training
   task, keep trajectories whose patch passes `local_eval.py`, retrain. These are
   on-policy and include real exploration, so they are better data than the scripted ones.
   Since the test repos are private, optimise for general behaviour and not for
   fastapi/rich/requests/httpx specifics.

## Step by step

All of this runs on Kaggle, since the data, model and L4 GPUs live there.

```bash
# 0. local sanity checks (any machine)
python tools/build_submission.py --out submission.zip      # validates + zips

# 1. Kaggle notebook (L4x4, internet OFF), attach competition data + Gemma 4 model
python training/build_sft_data.py --tasks $DATA/tasks.jsonl --snapshots $DATA/snapshots \
       --out training/data/sft.jsonl --holdout 20
python training/train_lora.py --model $GEMMA4_31B_IT --data training/data/sft.jsonl \
       --out /kaggle/working/main_lora
cp -r /kaggle/working/main_lora submission/adapters/main_lora
#    then uncomment `adapter: main_lora` in submission/agent.yaml

# 2. evaluate on the holdout with the harness CLI (see HARNESS_README.md), then
python tools/local_eval.py --data $DATA --gold --ids training/data/holdout_ids.txt  # harness check
python tools/local_eval.py --data $DATA --pred submission.parquet --ids training/data/holdout_ids.txt
```

## Assumptions to check against `HARNESS_README.md` / `sample_submission/`

These files weren't reachable from the environment this was written in.

- Field name for adapters (`adapter: main_lora`), and whether the harness tools must be
  listed under `tools:` by bare name (as here) or are injected automatically.
- How a sub-agent is attached (`AgentTool` + `config_path`), and whether
  `!include ../prompts/analyzer.md` from `sub_agents/` is allowed (it stays inside the
  root, but a strict validator may reject any `..`).
- Skill names: ADK requires `name` == directory name; kebab-case is always valid.
- `eval_config.yaml` format for per-task time/step limits.
- Whether the served model accepts LoRA target modules on the MLP projections.
