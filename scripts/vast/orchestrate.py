"""Run training jobs on vast.ai without anyone watching a terminal.

  launch   package the current commit, pick a filtered offer, start the job in args mode (the
           container exits when the job ends), watch its log, destroy the instance on probe
           failure / job end / timeout, retry on another host if the probe fails
           (with --eval-runs: a V* evaluation job on the runs' checkpoints instead, see eval_vstar.py)
  resume   watch a job of the ledger again (same --timeout-min and --budget as its launch)
  status   show the spend ledger and our running instances
  destroy-all  destroy every instance this tool created

The vast.ai API key is read from VAST_API_KEY and stays on this machine; rented hosts only get
HF_TOKEN and HF_RUNS_REPO. Spend is tracked in a ledger (hourly price x runtime + traffic) and
launch refuses to start when the worst-case cost of a job would exceed the remaining budget.

Example (smoke test on a cheap card):
  python scripts/vast/orchestrate.py launch --name smoke1 --gpu "RTX 4090" --min-cores 6 --min-ram-gb 32 \
      --config configs/ablations/smoke_qwen3_vl_2b.yaml --max-steps 20 --timeout-min 40 --budget 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offers import fetch, filter_offers, summary  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE = Path(os.environ.get("VAST_STATE", REPO_ROOT / "vast_state.json"))
LABEL_PREFIX = "spmc-"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
PROBE_DEFAULTS = {  # gpu name: (min bf16 TFLOPS, min power limit W)
    "RTX PRO 6000": (150, 540), "RTX 4090": (110, 405), "RTX 3090": (45, 315),
    "H100": (400, 630), "A100": (180, 230), "RTX 5090": (130, 520),
}
BOOTSTRAP = (
    # -U: the image may ship an older hub/hf_xet that pip would otherwise keep; <1.0 matches transformers<5
    "pip install -q -U 'huggingface_hub<1.0' hf_xet && "
    "python -c \"import os, tarfile; from huggingface_hub import hf_hub_download; "
    "p = hf_hub_download(os.environ['HF_RUNS_REPO'], os.environ['CODE_TARBALL']); "
    "os.makedirs('/root/attn_ft', exist_ok=True); tarfile.open(p).extractall('/root/attn_ft')\" && "
    "bash /root/attn_ft/scripts/vast/job.sh"
)


def vast(*args: str) -> str:
    key = os.environ["VAST_API_KEY"]
    # the key goes before the subcommand: `create instance ... --args` consumes every later token
    out = subprocess.run(["vastai", "--api-key", key, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"vastai {args[:2]} failed: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": [], "blacklist": []}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1))


def spent(state: dict) -> float:
    total = 0.0
    for job in state["jobs"]:
        end = job.get("ended") or time.time()
        total += job["dph"] * (end - job["created"]) / 3600 + job.get("traffic_usd", 0)
    return total


def package_code(repo: str, dry_run: bool) -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("commit your changes first: the job runs the committed code")
    path_in_repo = f"code/{sha}.tar.gz"
    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        if path_in_repo not in api.list_repo_files(repo):
            tarball = Path(f"/tmp/attn_ft-{sha}.tar.gz")
            subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(tarball), sha], cwd=REPO_ROOT, check=True)
            api.upload_file(path_or_fileobj=str(tarball), path_in_repo=path_in_repo, repo_id=repo,
                            commit_message=f"code {sha[:8]}")
    return path_in_repo


def probe_env(gpu: str) -> dict[str, str]:
    # the PROBE_DEFAULTS keys are substrings of both the vast.ai name and the nvidia-smi name
    key = next((k for k in PROBE_DEFAULTS if k in gpu), None)
    tflops, power = PROBE_DEFAULTS.get(key, (0, 0))
    env = {"PROBE_MIN_TFLOPS": str(tflops), "PROBE_MIN_POWER_W": str(power),
           "PROBE_MIN_DOWNLOAD_MBS": "40", "PROBE_MIN_DISK_MBS": "200"}
    if key:
        env["PROBE_GPU_SUBSTR"] = key
    return env


def create(offer: dict, env: dict[str, str], name: str, disk: int, dry_run: bool) -> int | None:
    env_str = " ".join(f"-e {k}={v}" for k, v in env.items())
    cmd = ["create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(disk), "--label",
           LABEL_PREFIX + name, "--env", env_str, "--cancel-unavail", "--raw",
           "--entrypoint", "bash", "--args", "-c", BOOTSTRAP]
    if dry_run:
        shown = [c if "HF_TOKEN=" not in c else "<env with tokens>" for c in cmd]
        print("DRY RUN: vastai " + " ".join(json.dumps(c) if " " in c else c for c in shown))
        return None
    text = vast(*cmd)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"create returned: {text.strip()[:300]}") from None
    if not out.get("success"):
        raise RuntimeError(f"create failed: {out}")
    return int(out["new_contract"])


def instance(iid: int) -> dict:
    return json.loads(vast("show", "instance", str(iid), "--raw"))


def logs(iid: int) -> str:
    try:
        return vast("logs", str(iid), "--tail", "400")
    except RuntimeError:
        return ""


def destroy(iid: int) -> None:
    # -y: without it the CLI waits for a confirmation nobody gives and the instance keeps billing
    for attempt in range(5):
        try:
            vast("destroy", "instance", str(iid), "-y")
        except RuntimeError as exc:
            print(f"destroy {iid} (attempt {attempt + 1}): {exc}")
        try:
            if iid not in [inst["id"] for inst in json.loads(vast("show", "instances", "--raw"))]:
                return
        except RuntimeError as exc:
            print(f"show instances: {exc}")
        time.sleep(10 * (attempt + 1))
    print(f"WARNING: instance {iid} still exists after destroy; run destroy-all")


def watch(job: dict, state: dict, timeout_min: int, budget: float) -> str:
    """Poll until the job ends; returns the outcome and destroys the instance."""
    iid, created = job["instance"], job["created"]
    probe_seen = started = False
    while True:
        time.sleep(60)
        text = logs(iid)
        info = {}
        try:
            info = instance(iid)
        except RuntimeError as exc:
            print(f"show instance {iid}: {exc}")
        status = info.get("actual_status")
        elapsed = (time.time() - created) / 60
        started = started or "JOB_START" in text
        for line in text.splitlines():
            if line.startswith("PROBE_RESULT") and not probe_seen:
                probe_seen = True
                job["probe"] = json.loads(line.split(" ", 1)[1])
                print(f"[{job['name']}] probe: {job['probe']}")
        outcome = None
        if "PROBE_FAIL" in text:
            outcome = "probe_failed"
        elif "JOB_END" in text:
            outcome = text[text.rfind("JOB_END"):].split("\n")[0]
        elif status in ("exited", "offline"):
            outcome = f"container_{status}"
        elif status != "running" and elapsed > 20:
            outcome = "never_started"
        elif status == "running" and not started and elapsed > 12:
            # the bootstrap (code download from the hub) failed and the container keeps restarting
            outcome = "bootstrap_failed"
        elif elapsed > timeout_min + 30:
            outcome = "orchestrator_timeout"
        elif spent(state) > budget:
            outcome = "budget_exceeded"
        if outcome:
            destroy(iid)
            job["ended"], job["outcome"] = time.time(), outcome
            save_state(state)
            print(f"[{job['name']}] {outcome} after {elapsed:.0f} min; total spent ${spent(state):.2f}")
            return outcome
        print(f"[{job['name']}] {status} {elapsed:.0f} min, spent ${spent(state):.2f}", flush=True)


def launch(args: argparse.Namespace) -> None:
    state = load_state()
    repo = os.environ["HF_RUNS_REPO"]
    tarball = package_code(repo, args.dry_run)
    skipped = set()  # offers that could not be rented (taken in the meantime)
    for attempt in range(args.max_attempts):
        offers, _ = filter_offers(fetch(args.gpu), args.min_cores, args.min_ram_gb)
        offers = [o for o in offers if o["machine_id"] not in state["blacklist"] and o["id"] not in skipped]
        if not offers:
            raise SystemExit(f"no acceptable {args.gpu} offer right now")
        offer = offers[0]
        worst_case = offer["dph_total"] * (args.timeout_min + 30) / 60 + offer["traffic_usd"]
        if spent(state) + worst_case > args.budget:
            raise SystemExit(f"worst case ${worst_case:.2f} would exceed the budget "
                             f"(${spent(state):.2f} of ${args.budget:.2f} spent)")
        print(f"attempt {attempt + 1}: {summary(offer)}")
        env = {"HF_TOKEN": os.environ["HF_TOKEN"], "HF_RUNS_REPO": repo, "CODE_TARBALL": tarball,
               "JOB_NAME": args.name, "JOB_CONFIG": args.config, "JOB_TIMEOUT_MIN": str(args.timeout_min),
               **probe_env(args.gpu)}
        if args.max_steps:
            env["ATTN_FT_MAX_STEPS"] = str(args.max_steps)
        if args.eval_runs:
            env["JOB_EVAL_RUNS"] = args.eval_runs
        if args.eval_steps:
            env["JOB_EVAL_STEPS"] = args.eval_steps
        try:
            iid = create(offer, env, args.name, args.disk, args.dry_run)
        except RuntimeError as exc:
            print(f"attempt {attempt + 1}: {exc}")
            skipped.add(offer["id"])
            # in case the instance was created after all
            for inst in json.loads(vast("show", "instances", "--raw")):
                if inst.get("label") == LABEL_PREFIX + args.name:
                    destroy(inst["id"])
            continue
        if iid is None:
            return
        job = {"name": args.name, "instance": iid, "machine": offer["machine_id"], "dph": offer["dph_total"],
               "traffic_usd": offer["traffic_usd"], "created": time.time(), "config": args.config}
        state["jobs"].append(job)
        save_state(state)
        try:
            outcome = watch(job, state, args.timeout_min, args.budget)
        except BaseException:
            # never leave a billing instance behind when the watcher itself dies
            destroy(iid)
            job["ended"], job["outcome"] = time.time(), "orchestrator_error"
            save_state(state)
            raise
        if outcome in ("probe_failed", "never_started", "bootstrap_failed"):
            state["blacklist"].append(offer["machine_id"])
            save_state(state)
            continue
        return
    raise SystemExit("gave up after repeated probe failures")


def resume(args: argparse.Namespace) -> None:
    """Watch a job of the ledger whose watcher was stopped (e.g. to move it to a detached process)."""
    state = load_state()
    job = next((j for j in reversed(state["jobs"]) if j["name"] == args.name and not j.get("ended")), None)
    if job is None:
        raise SystemExit(f"no running job named {args.name} in {STATE}")
    try:
        watch(job, state, args.timeout_min, args.budget)
    except BaseException:
        destroy(job["instance"])
        job["ended"], job["outcome"] = time.time(), "orchestrator_error"
        save_state(state)
        raise


def status(_: argparse.Namespace) -> None:
    state = load_state()
    for job in state["jobs"]:
        print(json.dumps({k: job.get(k) for k in ("name", "instance", "dph", "outcome", "probe")}))
    print(f"spent ${spent(state):.2f}; blacklisted machines: {state['blacklist']}")
    for inst in json.loads(vast("show", "instances", "--raw")):
        if str(inst.get("label", "")).startswith(LABEL_PREFIX):
            print(f"live: {inst['id']} {inst['label']} {inst.get('actual_status')} ${inst.get('dph_total')}/h")


def destroy_all(_: argparse.Namespace) -> None:
    for inst in json.loads(vast("show", "instances", "--raw")):
        if str(inst.get("label", "")).startswith(LABEL_PREFIX):
            destroy(inst["id"])
            print(f"destroyed {inst['id']} {inst['label']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", default="RTX PRO 6000 WS")
    p.add_argument("--min-cores", type=int, default=8)
    p.add_argument("--min-ram-gb", type=int, default=64)
    p.add_argument("--disk", type=int, default=100)
    p.add_argument("--timeout-min", type=int, default=240)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--eval-runs", help="V* evaluation job for these run names instead of training (eval_vstar.py)")
    p.add_argument("--eval-steps", help="comma-separated checkpoint steps (default 2000,4000,6000,8000 + last)")
    p.add_argument("--budget", type=float, required=True, help="total USD across all jobs in the ledger")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=launch)
    p = sub.add_parser("resume")
    p.add_argument("--name", required=True)
    p.add_argument("--timeout-min", type=int, default=240)
    p.add_argument("--budget", type=float, required=True)
    p.set_defaults(func=resume)
    sub.add_parser("status").set_defaults(func=status)
    sub.add_parser("destroy-all").set_defaults(func=destroy_all)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
