#!/usr/bin/env python3
"""Score predicted patches locally, mimicking the competition's Phase-2 grading.

For each task: unpack the snapshot, run the dataset's sandbox/setup.py
(editable install from /wheels), apply the predicted patch, apply the task's
test_patch, then run pytest on the test files that test_patch touches.
PASS == pytest exit code 0.

Runs inside the competition sandbox image so dependencies match:
  docker build -f docker/Dockerfile.sandbox -t swebench-sandbox:latest <dataset dir>

Usage:
  # sanity check the harness with the reference fixes (should be ~100%)
  python tools/local_eval.py --data /path/to/dataset --gold --ids training/data/holdout_ids.txt
  # score agent output (submission.parquet or jsonl with id/prediction)
  python tools/local_eval.py --data /path/to/dataset --pred /kaggle/working/submission.parquet
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile


def load_preds(path):
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return dict(zip(df["id"], df["prediction"]))
    return {r["id"]: r["prediction"] for r in map(json.loads, open(path)) if r}


def test_files(test_patch):
    return sorted({m for m in re.findall(r"^\+\+\+ b/(\S+)", test_patch, re.M) if m.endswith(".py")})


SCRIPT = r"""
set -e
mkdir -p /workspace && cd /workspace
tar xzf /snap.tgz -C /workspace
top=$(ls -A /workspace); if [ $(echo "$top" | wc -l) = 1 ] && [ ! -d .git ]; then shopt -s dotglob; mv "$top"/* . ; rmdir "$top"; fi
python /setup.py >/tmp/setup.log 2>&1 || { echo SETUP_FAILED; tail -20 /tmp/setup.log; exit 3; }
if [ -s /pred.patch ]; then git apply --whitespace=nowarn /pred.patch || { echo PRED_APPLY_FAILED; exit 4; }; fi
git apply --whitespace=nowarn /test.patch || { echo TEST_APPLY_FAILED; exit 5; }
timeout {timeout} python -m pytest -q -p no:cacheprovider {tests} 2>&1 | tail -15
exit ${PIPESTATUS[0]}
"""


def run_task(task, pred, data, image, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "pred.patch").write_text("" if pred in (None, "", "NO_PATCH") else pred)
        (t / "test.patch").write_text(task["test_patch"])
        tests = " ".join(test_files(task["test_patch"])) or "."
        (t / "run.sh").write_text(SCRIPT.replace("{timeout}", str(timeout)).replace("{tests}", tests))
        cmd = ["docker", "run", "--rm", "--network", "none",
               "-v", f"{data}/snapshots/{task['instance_id']}.tgz:/snap.tgz:ro",
               "-v", f"{data}/wheels:/wheels:ro",
               "-v", f"{data}/sandbox/setup.py:/setup.py:ro",
               "-v", f"{t}/pred.patch:/pred.patch:ro", "-v", f"{t}/test.patch:/test.patch:ro",
               "-v", f"{t}/run.sh:/run.sh:ro", image, "bash", "/run.sh"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 600)
            return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="competition dataset dir (tasks.jsonl, snapshots/, wheels/, sandbox/)")
    ap.add_argument("--pred", help="submission.parquet or jsonl with id/prediction")
    ap.add_argument("--gold", action="store_true", help="score the reference patches instead")
    ap.add_argument("--ids", help="file with instance ids to restrict to")
    ap.add_argument("--image", default="swebench-sandbox:latest")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="eval_results.jsonl")
    a = ap.parse_args()
    data = str(pathlib.Path(a.data).resolve())
    tasks = [json.loads(l) for l in open(f"{data}/tasks.jsonl") if l.strip()]
    if a.ids:
        keep = set(open(a.ids).read().split())
        tasks = [t for t in tasks if t["instance_id"] in keep]
    preds = {t["instance_id"]: t["patch"] for t in tasks} if a.gold else load_preds(a.pred)
    passed = 0
    with open(a.out, "w") as fh:
        for i, task in enumerate(tasks, 1):
            ok, log = run_task(task, preds.get(task["instance_id"]), data, a.image, a.timeout)
            passed += ok
            fh.write(json.dumps({"id": task["instance_id"], "pass": ok, "log": log}) + "\n")
            print(f"[{i}/{len(tasks)}] {task['instance_id']}: {'PASS' if ok else 'FAIL'}  (running {passed}/{i})", flush=True)
    print(f"SCORE: {passed}/{len(tasks)} = {100 * passed / max(1, len(tasks)):.1f}%")


if __name__ == "__main__":
    sys.exit(main())
