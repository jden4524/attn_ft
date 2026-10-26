#!/usr/bin/env bash
# Runs inside a rented vast.ai container (args launch mode): probe the machine, install the
# training dependencies, run one training config under a timeout, upload the job log, and exit.
# The container stops when this script exits, which stops GPU billing. vast.ai may restart an
# exited container, so a finished job leaves a marker and asks the host to stop the instance
# (CONTAINER_ID / CONTAINER_API_KEY are set by vast.ai); the orchestrator destroys it as well.
#
# Environment: HF_TOKEN, HF_RUNS_REPO, JOB_NAME, JOB_CONFIG, JOB_TIMEOUT_MIN,
# optional ATTN_FT_MAX_STEPS, JOB_EVAL_RUNS / JOB_EVAL_STEPS (see eval_vstar.py) and PROBE_* thresholds
# (see probe.py).
set -uo pipefail
cd /root/attn_ft
LOG=/root/job.log
exec > >(tee -a "$LOG") 2>&1

upload_log() {
    python - <<'EOF'
import os
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/root/job.log", repo_id=os.environ["HF_RUNS_REPO"],
                    path_in_repo=f"jobs/{os.environ['JOB_NAME']}/job.log",
                    commit_message=f"job log: {os.environ['JOB_NAME']}")
EOF
}

finish() {  # finish <status> <exit code>
    echo "JOB_END status=$1 rc=$2 $(date -u +%FT%TZ)"
    upload_log
    touch /root/job_done
    stop_self
    exit "$2"
}

stop_self() {
    [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] || return 0
    pip install -q vastai >/dev/null 2>&1
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" || echo "self-stop failed"
}

if [ -e /root/job_done ]; then
    echo "JOB_RESTARTED: the job already ended; stopping the instance"
    stop_self
    exit 0
fi

echo "JOB_START $(date -u +%FT%TZ) name=$JOB_NAME config=$JOB_CONFIG"
export HF_XET_HIGH_PERFORMANCE=1 WANDB_MODE=disabled PYTHONPATH=/root/attn_ft/src

python scripts/vast/probe.py || finish probe_failed 3

pip install -q "transformers<5.0.0" "datasets<4.0.0" accelerate peft pyyaml wandb torchvision || {
    finish install_failed 4; }

if [ -n "${JOB_EVAL_RUNS:-}" ]; then  # evaluation job: V* on the checkpoints of these runs
    timeout "${JOB_TIMEOUT_MIN}m" python scripts/vast/eval_vstar.py
else
    timeout "${JOB_TIMEOUT_MIN}m" python src/attn_ft/train.py --config "$JOB_CONFIG"
fi
rc=$?
if [ "$rc" -eq 0 ]; then status=ok; elif [ "$rc" -eq 124 ]; then status=timeout; else status=failed; fi
finish "$status" "$rc"
