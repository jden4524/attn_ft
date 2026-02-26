#!/usr/bin/env bash
set -euo pipefail

/venv/main/bin/python -m venv attn_ft
source attn_ft/bin/activate
uv pip install -e .