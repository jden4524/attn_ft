"""V* mechanistic evaluation of trained checkpoints, run inside a vast.ai job (JOB_EVAL_RUNS set).

For each run name in JOB_EVAL_RUNS (comma-separated `run_name`s of the training configs, or `base` /
`base_var` for the base model without / with VAR) this scores the checkpoints of the fixed protocol
(steps 2000, 4000, 6000, 8000 and the last one; override with JOB_EVAL_STEPS) with
eval/vstar_grounding.py and uploads each result to results/vstar/<run>/step-<n>.jsonl in
HF_RUNS_REPO. Results already in the repo are skipped, so a rerun continues where one stopped.
The heads come from `fixed_heads` of JOB_CONFIG. JOB_EVAL_LIMIT (testing) scores only the first N
questions and uploads nothing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    repo = os.environ["HF_RUNS_REPO"]
    runs = [r.strip() for r in os.environ["JOB_EVAL_RUNS"].split(",") if r.strip()]
    wanted = [int(s) for s in os.environ.get("JOB_EVAL_STEPS", "2000,4000,6000,8000").split(",")]
    limit = os.environ.get("JOB_EVAL_LIMIT")
    api = HfApi()
    files = set(api.list_repo_files(repo))

    heads = yaml.safe_load(open(ROOT / os.environ["JOB_CONFIG"]))["model"]["fixed_heads"]
    heads_path = Path(os.environ.get("JOB_EVAL_DIR", "/root")) / "heads.json"
    heads_path.write_text(json.dumps({str(k): v for k, v in heads.items()}))
    work = Path(os.environ.get("JOB_EVAL_DIR", "/root"))
    vstar = snapshot_download("craigwu/vstar_bench", repo_type="dataset", local_dir=str(work / "vstar_bench"))
    out_dir = work / "vstar_out"
    out_dir.mkdir(exist_ok=True)

    failed = []
    for run in runs:
        if run in ("base", "base_var"):
            jobs = [("base", None)]
        else:
            steps = sorted({int(m.group(1)) for f in files
                            if (m := re.fullmatch(rf"checkpoints/{re.escape(run)}/step-(\d+)/adapter_config\.json", f))})
            if not steps:
                print(f"EVAL_SKIP {run}: no checkpoints in {repo}", flush=True)
                failed.append(run)
                continue
            chosen = sorted({s for s in steps if s in wanted} | {steps[-1]})
            jobs = [(f"step-{s}", f"checkpoints/{run}/step-{s}") for s in chosen]
        for tag, adapter in jobs:
            target = f"results/vstar/{run}/{tag}.jsonl"
            if target in files:
                print(f"EVAL_DONE_ALREADY {target}", flush=True)
                continue
            out = out_dir / f"{run}_{tag}.jsonl"
            cmd = [sys.executable, str(ROOT / "eval/vstar_grounding.py"), "--heads", str(heads_path),
                   "--vstar-dir", vstar, "--out", str(out)]
            if adapter:
                cmd += ["--adapter", adapter]
            if run == "base_var":
                cmd.append("--var")
            if limit:
                cmd += ["--limit", limit]
            print(f"EVAL_START {run} {tag}", flush=True)
            rc = subprocess.run(cmd).returncode
            n = sum(1 for _ in open(out)) if out.exists() else 0
            if rc != 0:
                failed.append(f"{run}/{tag}")
            if rc == 0 and n and not limit:  # a partial file would count as done on the next run
                api.upload_file(path_or_fileobj=str(out), path_in_repo=target, repo_id=repo,
                                commit_message=f"V* eval {run} {tag} ({n} questions)")
            print(f"EVAL_END {run} {tag} rc={rc} questions={n}", flush=True)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
