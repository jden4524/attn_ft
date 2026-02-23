#!/usr/bin/env bash
set -euo pipefail

python -m venv attn_ft
source attn_ft/bin/activate
pip install -e .
