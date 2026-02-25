#!/usr/bin/env bash
set -euo pipefail

/opt/miniforge3/bin/python -m venv attn_ft
source attn_ft/bin/activate
pip install -e .
