#!/usr/bin/env bash

set -euo pipefail

GPU_COUNT=$(python - <<'PY'
import torch

count = torch.cuda.device_count()
if count <= 0:
	raise SystemExit("No CUDA GPUs detected.")
print(count)
PY
)

torchrun --nproc-per-node="$GPU_COUNT" VLMEvalKit/run.py --config eval_config_memvr.json --work-dir eval_results