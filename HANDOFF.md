# Handoff: controlled 2B study (branches `claude/nice-sagan-qr0cy9`, then `claude/awesome-lamport-pur4u7`)

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

The smoke test passed on 2026-10-26 (`smoke3`: probe, 20 steps, checkpoints `step-10`/`step-20`, destroy) after these fixes:
the API key went after `--args` (it would have been passed into the container), `destroy` needed `-y`, the image's
preinstalled hub broke the probe download (bootstrap now upgrades `huggingface_hub<1.0` and `hf_xet`), hosts in
mainland China cannot reach the hub (filtered out; a job that never prints `JOB_START` is retried elsewhere),
an offer that is gone at create time is skipped. vast.ai restarts exited containers, so `job.sh` leaves a marker
and stops its own instance. Smoke throughput on an RTX 4090: about 12 samples/s; a full epoch is about 10.4k steps
of 32, so use `--timeout-min 300` on an RTX PRO 6000 (`"RTX PRO 6000 WS"` or `"RTX PRO 6000 S"`).
`orchestrate.py launch` blocks while it watches a job; run it in the background for long jobs.

## V* evaluation
`eval/vstar_grounding.py` (session-1 script plus adapter loading, option scoring over the "A"/" A"/"(A" forms,
answer-format mass, knockout margins, crop hit rate, `--var`) and `eval/vstar_compare.py` (paired comparisons).
On a GPU host: `orchestrate.py launch --name vstar1 --gpu "RTX 4090" --min-cores 6 --min-ram-gb 32
--config configs/ablations/qwen3_vl_2b_spmc.yaml --eval-runs Qwen3-VL-2B_spmc_reference,Qwen3-VL-2B_autor_lateLoRA,Qwen3-VL-2B_spmc_shuffled_masks
--timeout-min 120 --budget <total>` writes `results/vstar/<run_name>/step-<n>.jsonl` to the runs repo.
Base and base+VAR were scored on CPU in session 2 (`results/vstar/base`, `results/vstar/base_var`).

## Session 2 status (2026-10-26)
All runs and V* evaluations of the fixed protocol are done; no vast.ai instance is left running.
- Trained (checkpoints every 1,000 steps + final in `checkpoints/<run_name>/` of the runs repo): `Qwen3-VL-2B_spmc_reference`,
  `Qwen3-VL-2B_autor_lateLoRA`, `Qwen3-VL-2B_spmc_shuffled_masks`, `Qwen3-VL-2B_spmc_squared_loss`, `Qwen3-VL-2B_a3tune_core`
  (`configs/ablations/qwen3_vl_2b_a3tune_core.yaml`, `loss: a3tune_core`, tests in `tests/test_a3tune_loss.py`).
- V* results per step: `results/vstar/<run_name>/step-<n>.jsonl` (GPU), base and base+VAR in `results/vstar/base*`.
- Write-ups: `session2/RESULTS_vstar_trio.md` and `session2/RESULTS_vstar_a3tune.md` in the runs repo; the training-tuple
  mass profile (`eval/train_mass_profile.py`) in `results/train_mass_profile/`.
- The interleave of the two datasets is unseeded, so sample order (and the step count, 10,355-10,397) differs slightly per run.
- The first a3tune_core attempt died on a device-side assert in the LM cross-entropy (label out of range); `train.py` now masks
  and reports such labels. The rerun on another host flagged none.

## Pending
- Fill the rebuttal drafts from the session-2 results (coordinate with any session editing `session1/rebuttal/rebuttal_v4.md`).
- Optional: box_masks run.
