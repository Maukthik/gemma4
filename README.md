# Gemma 4 Developer Agent Competition: submission workspace

Goal: a `submission.zip` holding an ADK agent config that drives
`gemma-4-31b-it-qat-w4a16-ct` (optionally with LoRA adapters). It must fix
~120 hidden Python issues offline. Score = share of tasks whose hidden tests pass.

## Facts that shape the design

Verified against the official harness source (swegemma 0.2.7, adk-submission 0.2.11,
google-adk 1.36.1); scores are other teams' public reports.

| Fact | Consequence here |
|---|---|
| Tasks run **sequentially**; going over **12 h fails the whole submission** | `eval_config.yaml`: 4.5 min / 40 calls / 80 turns per task (~11 h worst case) |
| The sample `eval_config.yaml` allows 1 min / 10 calls | that's why early scores were ~0; never reuse it |
| vLLM `max_model_len=32768`, **no compaction inside a task** | the prompt enforces short outputs and a ~20-call plan; `max_output_tokens: 4096` |
| `submit_patch` is free and can be repeated; the session ends on a text-only reply after submitting | the prompt says: submit as soon as the fix is in, resubmit after changes, finish with one sentence |
| `write_file("/tmp/x")` writes `/workspace/tmp/x`; untracked files end up in the patch | scratch goes to `/tmp` via `run_command` heredoc only |
| Test files are reset before grading; `pytest.ini`/`conftest.py` are harness files | never edit tests or harness files |
| `instruction` is ADK-templated | `{problem_description}` restates the issue; any other `{word}` would crash, and the validator checks for it |
| Schema: `tools` = names or `agent_tool: {config_path, skip_summarization}`; `skills:` = paths; no `..` in `config_path` | `agent.yaml` uses exactly that |
| Thinking: `include_thoughts: false` → `enable_thinking=false` | thinking off (every public 0.10–0.12 entry does this) |
| LoRA served with `max_lora_rank=128`; rank-32 all-layer was reported unstable on W4A16 | default LoRA rank 8, attention only |
| Public leaderboard ~0.10–0.15; prompt tweaks plateau ~0.10 | **training is the lever**: SFT, then rejection-sampling SFT |

## Layout

```
submission/                     -> submission.zip (agent.yaml at the root)
  agent.yaml, eval_config.yaml  root agent (8 tools + 2 skills), per-task budgets
  configs/sampling.yaml         temp 0.2, top_k 40, 4096 tokens, thinking off
  prompts/system.md             budget-aware workflow (locate -> repro in /tmp -> fix -> submit -> verify)
  prompts/analyzer.md, sub_agents/code_analyzer.yaml   optional read-only helper (off by default)
  skills/repo-navigation/       find_symbol.py, find_tests.py, repo_map.py + checklists
  skills/verify-fix/            check_patch.py (scratch/pycache/test edits/syntax), run_tests.py
  adapters/                     trained PEFT adapters go here (main_lora/)
tools/build_submission.py       validator mirroring adk-submission rules + zip
tools/local_eval.py             light re-scorer for existing patches (grader-like pytest)
training/build_sft_data.py      scripted expert trajectories replayed on real snapshots, harness-format observations
training/traces_to_sft.py       official ATIF traces of *resolved* runs -> SFT data (rejection sampling)
training/train_lora.py          QLoRA SFT, assistant-only loss, multiple --data files
kaggle/run_agent_eval.py        official harness on Kaggle L4x4: evaluate, or sample rollouts for training
```

Verified here: the submission compiles with the real `adk_submission.compile_submission`
(google-adk 1.36.1), all tools resolve, skills load, and skill scripts run through ADK's
`run_skill_script`. The SFT generator, the trace converter (tested on real `adk_eval_core` ATIF
output) and the loss masking all have tests. **Not yet run:** anything needing the GPU,
model or dataset.

## Plan

1. **Baseline:** `python tools/build_submission.py`, then submit `submission.zip` and get a score.
2. **Holdout measurement** (Kaggle notebook, L4x4, internet off; attach the competition data, the
   QAT model, and `metric/gemma-4-developer-agent-wheelhouse`):
   ```bash
   python training/build_sft_data.py --tasks $DATA/tasks.jsonl --snapshots $DATA/snapshots \
          --out training/data/sft.jsonl --holdout 30            # also writes holdout_ids.txt
   python kaggle/run_agent_eval.py --task-ids training/data/holdout_ids.txt --results-dir /kaggle/working/base
   ```
3. **Rollouts on the training tasks** (diverse sampling), keeping only resolved runs:
   ```bash
   python kaggle/run_agent_eval.py --exclude-ids training/data/holdout_ids.txt --repeat 4 \
          --temperature 0.7 --results-dir /kaggle/working/rollouts
   python training/traces_to_sft.py --runs /kaggle/working/rollouts/run_* \
          --exclude training/data/holdout_ids.txt --out training/data/rft.jsonl
   ```
4. **Train** `main_lora` on the mix, load-test it in vLLM, then re-measure the holdout:
   ```bash
   python training/train_lora.py --model $GEMMA4_31B_IT_BF16 --data training/data/sft.jsonl \
          --data training/data/rft.jsonl --out /kaggle/working/main_lora
   cp -r /kaggle/working/main_lora submission/adapters/main_lora   # uncomment `adapter: main_lora`
   ```
   Repeat steps 3–4 with the adapter enabled (expert iteration). Submit only if the holdout improves.
5. **A/B (one change at a time):** analyzer sub-agent on/off, skills on/off, 40 vs 48 tool calls.
   The holdout is only ~30 tasks, so treat ±3 tasks as noise.

## Open risks

- The scorer's per-task overhead isn't public: keep `max_time_minutes × 120` well under 12 h.
- The LoRA must load in vLLM against the compressed-tensors checkpoint; test before submitting.
- Training on the bf16 checkpoint while serving W4A16 QAT is standard, but measure it.
- Hidden repos are private, not fastapi/rich/requests/httpx, so avoid overfitting to those four.
