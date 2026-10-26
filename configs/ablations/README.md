# Qwen3-VL-2B ablation runs

All runs share the reference sPMC setup in `qwen3_vl_2b_spmc.yaml` (LoRA r=16 on q/k of the last 40% of layers,
the same 72 fixed heads, lambda=1, seed 42) and change one thing.
These runs form a standalone controlled study; they are not meant to reproduce the paper's numbers.

| Config | What changes |
|---|---|
| `qwen3_vl_2b_spmc.yaml` | Reference sPMC (SAM3 masks, adaptive late-layer heads) |
| `qwen3_vl_2b_autor_matched.yaml` | No attention loss (`loss_weight: 0`); LoRA on the same late layers as sPMC |
| `qwen3_vl_2b_spmc_shuffled_masks.yaml` | Each sample is supervised with the mask of a different image in its micro-batch |
| `qwen3_vl_2b_spmc_box_masks.yaml` | Each connected mask region is replaced by its bounding box |
| `qwen3_vl_2b_spmc_gsam2_masks.yaml` | Masks from Grounded-SAM 2 instead of SAM3 (build them first, see below) |
| `qwen3_vl_2b_spmc_squared_loss.yaml` | `(1 - p_R)^2` instead of `-log p_R`, same in-mask mass `p_R` |
| `qwen3_vl_2b_spmc_heads_x0.5.yaml` / `_x2.yaml` | Both selection thresholds moved so the selected set is about half / double its size |

`fixed_heads` holds the 72 heads selected on the base model, so every run supervises the same heads
and calibration is skipped. The `heads_x*` runs leave it unset and re-select (calibration always
uses the SAM3 masks, also when the run itself trains on shuffled or box masks).

Launch one run. `HF_RUNS_REPO` sends checkpoints to `checkpoints/<run_name>/step-<step>` in that repo;
`ATTN_FT_MAX_STEPS` caps the number of optimizer steps (for smoke tests):

```bash
HF_RUNS_REPO=<user>/<repo> CONFIG_PATH=configs/ablations/qwen3_vl_2b_spmc_shuffled_masks.yaml bash scripts/train.sh
```

Grounded-SAM 2 masks (one shard per GPU, then merge and push):

```bash
python scripts/build_gsam2_masks.py --src Jackie2235/RLAIF-V-SEG --out-dir gsam2/rlaif --shard 0 --num-shards 4
python scripts/build_gsam2_masks.py --merge gsam2/rlaif --push-to-hub <user>/RLAIF-V-SEG-gsam2
```

Rows where Grounded-SAM 2 finds nothing, or whose merged mask covers 60% or more of the image, are dropped
(`gsam2_stats.json` reports how many). If many rows are dropped, also train the reference on the
same subset so that the two runs differ only in the masks.
