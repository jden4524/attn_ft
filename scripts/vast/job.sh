#!/usr/bin/env bash
# Runs inside a rented vast.ai container (args launch mode): probe the machine, install the
# training dependencies, run one training config under a timeout, upload the job log, and exit.
# The container stops when this script exits, which stops GPU billing.
#
# Environment: HF_TOKEN, HF_RUNS_REPO, JOB_NAME, JOB_CONFIG, JOB_TIMEOUT_MIN,
# optional ATTN_FT_MAX_STEPS and PROBE_* thresholds (see probe.py).
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

echo "JOB_START $(date -u +%FT%TZ) name=$JOB_NAME config=$JOB_CONFIG"
export HF_XET_HIGH_PERFORMANCE=1 WANDB_MODE=disabled PYTHONPATH=/root/attn_ft/src

if ! python scripts/vast/probe.py; then
    echo "JOB_END status=probe_failed"
    upload_log
    exit 3
fi

pip install -q "transformers<5.0.0" "datasets<4.0.0" accelerate peft pyyaml wandb torchvision || {
    echo "JOB_END status=install_failed"; upload_log; exit 4; }

timeout "${JOB_TIMEOUT_MIN}m" python src/attn_ft/train.py --config "$JOB_CONFIG"
rc=$?
if [ "$rc" -eq 0 ]; then status=ok; elif [ "$rc" -eq 124 ]; then status=timeout; else status=failed; fi
echo "JOB_END status=$status rc=$rc $(date -u +%FT%TZ)"
upload_log
exit "$rc"
