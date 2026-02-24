#!/usr/bin/env bash
set -euo pipefail
source /etc/profile
source ~/.bashrc

python -m venv attn_ft
source attn_ft/bin/activate
pip install -e .
