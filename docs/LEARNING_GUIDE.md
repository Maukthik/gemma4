# A beginner's guide to this project

What this repo does, how it was built, and how to apply the same habits to your
own code. Read it top to bottom once, then use Part 7 for hands-on practice.

## Part 1: The big picture

**What the competition wants.** Google gives everyone one AI model, Gemma 4. You
turn it into a *coding agent*: a program that's handed a bug report, explores a
code repository on its own, edits the code to fix the bug, and hands in the fix.
Hidden tests then check whether the fix works. Your score is the percentage of
bugs fixed.

**An analogy.** Think of Gemma as a smart new intern who has never seen the codebase:

| Real thing | Intern analogy | File in this repo |
|---|---|---|
| System prompt | The instruction sheet you give the intern on day one | `submission/prompts/system.md` |
| Tools | What the intern may use: a terminal, a file viewer, an editor | listed in `submission/agent.yaml` |
| Skills | Small helper scripts written for the intern | `submission/skills/` |
| Sub-agent | A second intern who only researches and never edits | `submission/sub_agents/code_analyzer.yaml` |
| Budget | "You have 4.5 minutes per bug" | `submission/eval_config.yaml` |
| LoRA adapter | Extra training that changes the intern's habits | `submission/adapters/` (not trained yet) |
| Fine-tuning data | Worked examples of how to fix bugs | `training/` |

You don't write the agent's "brain"; Google supplies the model. You write the
**instructions, tools and training** that make the model behave well.

## Part 2: Key words

- **LLM (large language model):** a program that predicts text. Gemma is one.
- **Token:** a chunk of text, roughly ¾ of a word. Models read and write tokens.
- **Context window:** how much text the model can "see" at once. Here it's 32,768
  tokens, which is why the prompt tells the agent to keep command outputs short.
- **Prompt:** the text instructions given to the model.
- **Tool call:** the model writes something like `run_command("grep ...")`, the
  harness runs it, and the result is fed back to the model.
- **Harness:** the organizers' program that runs your agent, gives it tools,
  enforces time limits and grades the result.
- **Patch / diff:** a text file describing which lines changed. `git diff` produces one.
- **Fine-tuning / SFT (supervised fine-tuning):** showing the model examples
  ("here's a bug, here's how an expert solved it") so it copies that behaviour.
- **LoRA:** a cheap way to fine-tune. Instead of changing all 31 billion numbers in
  the model, you train a small add-on of a few million numbers.
- **Rejection sampling:** let the agent try many times, keep only the attempts
  that worked, and train on those.

## Part 3: A tour of the repo

```
gemma4/
├── submission/   ← what you upload (zipped into submission.zip)
├── tools/        ← helpers for YOU (validate, zip, score locally)
├── training/     ← make training data + train the LoRA
├── kaggle/       ← run the official evaluator on Kaggle GPUs
└── docs/         ← this guide
```

### `submission/`: the agent itself

- **`agent.yaml`** is the config file. YAML is just a readable way to write settings:
  ```yaml
  name: swe_coder
  model: gemma-4-31b-it-qat-w4a16-ct
  instruction: !include prompts/system.md    # "paste that file's text here"
  tools:
    - run_command
    - edit_file
  ```
- **`prompts/system.md`** is the instruction sheet. Every rule in it exists because
  of an observed failure:
  - "Keep outputs short": long outputs filled the context window and crashed runs.
  - "Put scratch files in `/tmp`": files left in `/workspace` ended up inside the
    patch and broke it.
  - "Call `submit_patch` early": agents that ran out of time before submitting lost
    their work.
- **`eval_config.yaml`** sets the time and call budget. The organizers' sample
  allowed only 1 minute per task, which is why most early scores were 0.
- **`skills/`** holds small Python scripts the agent can run. For example,
  `find_symbol.py` finds where a function is defined, and `check_patch.py` checks
  the patch for mistakes before submitting.

### `tools/`: for you, not the agent

- **`build_submission.py`** checks the submission against the competition's rules
  and zips it. It catches mistakes *before* you waste a daily submission.
- **`local_eval.py`** scores patches on your own machine, roughly the way the
  graders do.

### `training/`: making the model better

- **`build_sft_data.py`** takes each training task's known-correct fix and turns it
  into an "expert run": search, read the file, edit, submit. It runs every step on
  the real repository, so the outputs in the examples are real.
- **`traces_to_sft.py`** turns logs of the agent's own *successful* runs into
  training examples (rejection sampling).
- **`train_lora.py`** does the actual training on Kaggle's GPUs.

### `kaggle/run_agent_eval.py`

Runs the submission through the organizers' real evaluator on Kaggle, so you can
measure a score before submitting.

## Part 4: How the work was done (the most important lesson)

Writing code was the smaller part of the job. Most of the work was **finding out
what's true and then checking it.** These are the steps, with real examples from
building this repo.

### Step 1: Explore before building

The first commands were simple checks: is the repo empty? Is there internet? A
GPU? The answers (no Kaggle access, no GPU) decided what was possible before any
code was written.

### Step 2: State your guesses, then verify them

The first `agent.yaml` was a guess, and the README said so. Later the organizers'
real code turned up. Its schema file showed the guess was wrong in three places,
for example:

```yaml
tools:
  - name: run_command      # the guess: WRONG
  - run_command            # the real format
```

**Lesson:** when you can't check something, say so clearly. When you can, read
the actual source code rather than trusting memory or blog posts.

### Step 3: Test with small fake data

Without the real dataset, a tiny fake repo stood in: one file, two functions, one
bug fix. Running the training-data generator on it found **two real bugs**:

1. A temporary file (`gold.patch`) was written *inside* the repo, so it showed up
   in the patch.
2. Checking the code's syntax created `__pycache__/*.pyc` files, which also leaked
   into the patch.

The same leak would have hurt the real agent, so both the generator and
`check_patch.py` were fixed.

**Lesson:** a 10-line fake example finds bugs faster than staring at code. Always
run your code at least once on something small.

### Step 4: Test the failure cases too

The validator was fed a deliberately broken config to check that it complains:

```
ERROR: tools entries must be a tool name or `agent_tool: {config_path: ...}`
ERROR: config_path must be relative without '..': ../x.yaml
```

A checker that never fails might just be broken.

### Step 5: Learn from evidence, not opinions

A competitor's public notes reported results like "5.5 min per task → submission
failed (over 12 hours)" and "4.5 min → 0.10". Measured results like these became
settings. When their notes corrected an earlier claim, the correction won.

### Step 6: Write the "why" down

Comments in this repo explain *why*, not *what*:

```yaml
# timeout_seconds is the per-command timeout AND the hidden-test pytest timeout
# during verification, so keep it >= 180 or correct patches can time out.
timeout_seconds: 180
```

A future reader (you, in a month) can see what `180` is. The comment tells you
why you mustn't lower it.

## Part 5: How to read and understand code

Don't read a file top to bottom like a novel. Use this order:

1. **Read the docstring or top comment.** It tells you the file's purpose.
2. **Find `main()` or the entry point.** That's where the program starts.
3. **Follow the data.** What comes in, what changes it, what goes out?
4. **Look things up as you meet them.** Don't try to understand every function in advance.
5. **Run it and watch.** Add `print()` statements, or run it on a tiny example.

### Worked example: `submission/skills/verify-fix/scripts/check_patch.py`

```python
status = git("status", "--porcelain", "--untracked-files=all", cwd=cwd).splitlines()
files = [l[3:].strip() for l in status if l.strip()]
```

Read it piece by piece:

- `git status --porcelain` prints changed files in a machine-friendly format, like
  ` M pkg/util.py`.
- `.splitlines()` turns that text into a list, one line per file.
- `l[3:]` drops the first 3 characters (the status code, e.g. ` M `), leaving the
  file name.
- `if l.strip()` skips empty lines.

So `files` is "the list of changed file names." Then the script loops over them
and flags problems:

```python
if SCRATCH.search(f):
    errors.append(f"scratch file will be submitted: {f} ...")
```

`SCRATCH` is a regular expression (a text pattern) matching names like `repro.py`
or `debug.log`. Once you know what `files` holds, the rest reads like English.

**Try it:** open `submission/skills/repo-navigation/scripts/find_symbol.py` and
use the five steps above to answer: what does `--include-tests` change?

## Part 6: How to write better code

These principles are each used somewhere in this repo.

1. **One function, one job.** `build_sft_data.py` has `parse_patch()` (read a
   diff), `make_edits()` (turn it into edits) and `build_one()` (run a whole task).
   Each can be understood and tested alone.
2. **Use names that explain themselves.** `is_test_path(p)` needs no comment.
   `x2(p)` would.
3. **Check your inputs and fail loudly.** The validator stops with a clear message
   instead of letting a bad zip fail silently on Kaggle 12 hours later. A loud
   error now is cheaper than a mystery later.
4. **Verify your own output.** After replaying the edits, `build_sft_data.py`
   checks that the result matches the real fix *exactly*. Mismatching examples are
   skipped, not trusted:
   ```python
   if (root / p).read_text(encoding="utf-8") != txt:
       return None, f"replayed edits differ from gold for {p}"
   ```
5. **Start simple, then improve with evidence.** Version 1 was a guess, and
   version 2 fixed it using facts. Don't try to make version 1 perfect; make it
   runnable, then measure.
6. **Change one thing at a time.** If you change the prompt *and* the temperature
   *and* the budget, then see the score move, you won't know which change
   mattered. The README's A/B step says exactly this.
7. **Keep junk out of git.** `.gitignore` excludes `submission.zip`,
   `__pycache__` and the training data.
8. **Commit often, with messages that explain the change.** Commit messages here
   list *what* changed and *why*, so `git log` reads like a diary of the project.

## Part 7: Practice exercises (in order of difficulty)

Run these from the repo root (`gemma4/`).

### Exercise 1: run a skill script by hand

```bash
python3 submission/skills/repo-navigation/scripts/repo_map.py --root submission --include-tests
```

Expected output starts like this:

```
skills/repo-navigation/scripts/find_symbol.py: is_test(); main()
skills/repo-navigation/scripts/find_tests.py: main()
```

Explain each line: which file is it, and what are the names after the colon?
Then try `find_symbol.py --name main --root submission`.

### Exercise 2: break the validator on purpose

```bash
cp -r submission /tmp/broken_submission
echo 'Repo: {repo}' >> /tmp/broken_submission/prompts/system.md
python3 tools/build_submission.py --src /tmp/broken_submission --out /tmp/broken.zip
```

You should see:

```
ERROR: agent.yaml: instruction contains {repo}, which is not a session-state key ...
```

Find the code in `tools/build_submission.py` that produced this message (search
for `session-state key`). Read the function it's in and explain why `{repo}` is
rejected but `{problem_description}` is allowed.

### Exercise 3: add a small feature

Make `check_patch.py` print a WARNING when the diff adds more than 300 lines. A
huge patch for a small bug usually means something went wrong.

Hints:
- The script already counts added lines for its `DIFF LINES` output. Store that
  count in a variable instead of computing it inline.
- Append to the existing `warnings` list; it's already printed at the end.
- Test it: in a scratch git repo, add a 400-line file and run the script there.

### Exercise 4: write your first test

```bash
pip install pytest
mkdir -p tests
```

Create `tests/test_parse_patch.py`:

```python
import sys
sys.path.insert(0, "training")
from build_sft_data import parse_patch

PATCH = """diff --git a/pkg/util.py b/pkg/util.py
--- a/pkg/util.py
+++ b/pkg/util.py
@@ -1,3 +1,3 @@
 def f(a):
-    return a
+    return a or 0
"""

def test_parse_patch_finds_file_and_hunk():
    files = parse_patch(PATCH)
    assert list(files) == ["pkg/util.py"]
    hunk = files["pkg/util.py"][0]
    assert hunk["old_start"] == 1
    assert "    return a\n" in hunk["old"]
    assert "    return a or 0\n" in hunk["new"]
```

Run it with `python3 -m pytest tests/ -q`. Then make it fail on purpose (change
`== 1` to `== 2`) and read the error message pytest gives you. Learning to read
failures is half of testing.

### Exercise 5: improve the prompt with evidence

After your first Kaggle run, open a few failed traces in the results folder
(`traces/trace_<id>.json`). Find the most common mistake, add **one** rule to
`submission/prompts/system.md` that targets it, and measure again. Don't change
anything else in the same run.

## Part 8: What to learn next

1. **Python basics:** functions, lists, dicts, loops, reading files. Free option:
   the official Python tutorial at docs.python.org/3/tutorial.
2. **Git:** `status`, `add`, `commit`, `diff`, `log`, `branch`. Practise on this repo.
3. **The command line:** `ls`, `cd`, `grep`, `cat`, pipes (`|`). The agent uses
   exactly these.
4. **Testing with pytest:** write small tests for small functions (Exercise 4 is a
   start).
5. **The ML side:** what fine-tuning and LoRA are (the Hugging Face course is
   free), and how to read training logs.

The habit that matters most: **read before you write, run before you trust,
measure before you change.**
