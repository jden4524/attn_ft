#!/usr/bin/env bash
set -euo pipefail

source attn_ft/bin/activate
accelerate launch --multi_gpu src/attn_ft/train.py --config configs/qwen3_vl_8b_attn_ft_ce.yaml
accelerate launch --multi_gpu src/attn_ft/train.py --config configs/qwen3_vl_8b_attn_ft_vacuum.yaml