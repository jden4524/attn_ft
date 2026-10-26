# Handoff: controlled 2B study (branch `claude/nice-sagan-qr0cy9`)

Fuller context, results and decisions are in the private model repo `Jackie2235/spmc-rebuttal-runs`
under `session1/` (start with `session1/HANDOFF_PRIVATE.md`). That repo also receives code tarballs,
checkpoints (`checkpoints/<run>/step-<n>`) and job logs (`jobs/<name>/job.log`).

## What is on this branch
| Path | Purpose |
|---|---|
| `configs/ablations/` | One Qwen3-VL-2B config per condition; shared q/k LoRA on the last 40% of layers and 72 fixed heads. `smoke_qwen3_vl_2b.yaml` is a tiny-batch smoke test. |
| `src/attn_ft/data.py`, `losses.py`, `attn_hooks.py`, `train.py` | Mask modes (`sam`/`shuffled`/`box`), `(1-p)^2` loss form, head-set scaling, fixed heads, `HF_RUNS_REPO` upload target, `ATTN_FT_MAX_STEPS` cap |
| `src/attn_ft/var.py`, `eval/run_vlmeval.py` | VAR port for Qwen3-VL and a VLMEvalKit launcher (eager attention, needs `transformers<5`) |
| `scripts/build_gsam2_masks.py` | Grounded-SAM 2 mask regeneration |
| `scripts/vast/` | Offer filter, on-host probe, job script, orchestrator (see below) |

## Running on vast.ai
Environment of the orchestrating session: `VAST_API_KEY`, `HF_TOKEN` (write access to the runs repo),
`HF_RUNS_REPO=Jackie2235/spmc-rebuttal-runs`, the `vastai` CLI (`pip install vastai`).

1. `python scripts/vast/offers.py "RTX 4090" --min-cores 6 --min-ram-gb 32` shows what passes the filter.
2. Smoke test (about $0.5 worst case):
   `python scripts/vast/orchestrate.py launch --name smoke1 --gpu "RTX 4090" --min-cores 6 --min-ram-gb 32 --config configs/ablations/smoke_qwen3_vl_2b.yaml --max-steps 20 --timeout-min 40 --budget 5`
   Expect `PROBE_RESULT`, training steps, two checkpoints in `checkpoints/smoke_qwen3_vl_2b/`, `JOB_END status=ok`.
3. Training runs (about 2-2.5 h each on an RTX PRO 6000):
   `python scripts/vast/orchestrate.py launch --name <run> --gpu "RTX PRO 6000 WS" --config configs/ablations/<config>.yaml --timeout-min 240 --budget <total>`
4. `orchestrate.py status` and `orchestrate.py destroy-all` for bookkeeping and cleanup.

Not yet exercised against the live API: instance creation, log polling, destroy. Run the smoke test first.
`orchestrate.py launch` blocks while it watches a job; run it in the background for long jobs.

## Pending
- Smoke test, then the three runs: `qwen3_vl_2b_spmc`, `qwen3_vl_2b_autor_matched`, `qwen3_vl_2b_spmc_shuffled_masks`.
- Analyses of the trained adapters on held-out data (see the private handoff for the fixed protocol).
