"""In-mask attention mass on training tuples under the base model (or an adapter), and the gradient
weight each loss form gives at that mass.

For a fixed, seeded sample of training tuples (the two training datasets in size proportion, from
the first parquet shard of each), and the fixed heads of a config:
  * p_R per head and phrase token: softmax of the head's logits over the image tokens, mass inside
    the mask (paper Eq. 7); p_code is the mass the training loss uses (soft_suppression_loss with
    temp=2: logits are multiplied by the mask, so each out-of-mask token adds exp(0) to the in-mask
    sum and each in-mask token adds exp(0) to the out-of-mask sum, whose logits are halved);
  * p_a3: in-mask share of the A3Tune-style map (full-row softmax, mean over heads and tokens);
  * the push of each loss on the in-mask logits (derivative under a common shift of them):
    -log p -> (1 - p), (1 - p)^2 -> 2 p (1 - p)^2.
Writes one JSON line per tuple and prints a summary.

Usage: python eval/train_mass_profile.py --config configs/ablations/qwen3_vl_2b_spmc.yaml --n 200 --out out.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attn_ft.attn_hooks import AttnHookManager, extract_t2i_attn_valid, extract_t2i_rows  # noqa: E402
from attn_ft.data import AttnSupervisionCollator  # noqa: E402
from attn_ft.losses import a3tune_core_loss  # noqa: E402
from attn_ft.models import load_model_and_processor  # noqa: E402
from datasets import load_dataset  # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SIZES = {"Jackie2235/RLAIF-V-SEG": 61594, "Jackie2235/flickr30k_attn_ft": 271141}


def sample_tuples(n, seed):
    """n training tuples, split between the two datasets in proportion to their sizes (as train.py
    mixes them), drawn at random from the first parquet shard of each."""
    rng = np.random.default_rng(seed)
    total = sum(SIZES.values())
    picks = []
    for ds_id, size in SIZES.items():
        d = load_dataset(ds_id, split="train", data_files="data/train-00000-of-*.parquet",
                         verification_mode="no_checks")  # one shard, so the split size differs
        if "answer" not in d.column_names:
            d = d.rename_column("caption", "answer")
        k = round(n * size / total)
        picks += [(ds_id, int(j)) for j in rng.choice(len(d), size=k, replace=False)]
        SHARDS[ds_id] = d
    order = rng.permutation(len(picks))
    for j in order:
        ds_id, row = picks[j]
        sample = dict(SHARDS[ds_id][row])
        sample.setdefault("question", "Describe the image.")
        yield sample


SHARDS = {}


def in_mask_mass(logits, mask, temp=1.0):
    """logits [heads, tokens, n_img] over image tokens; mask [n_img] binary -> p [heads, tokens]."""
    m = mask.bool()
    pos = torch.logsumexp(logits[..., m], -1)
    neg = torch.logsumexp(logits[..., ~m] / temp, -1) if (~m).any() else torch.full_like(pos, -torch.inf)
    return torch.exp(pos - torch.logaddexp(pos, neg))


def code_mass(logits, mask, temp=2.0):
    """In-mask mass as soft_suppression_loss computes it -> p [heads, tokens]."""
    m = mask.bool()
    pos = torch.logsumexp(torch.where(m, logits, torch.zeros_like(logits)), -1)
    neg = torch.logsumexp(torch.where(m, torch.zeros_like(logits), logits / temp), -1)
    return torch.exp(pos - torch.logaddexp(pos, neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config whose fixed_heads are profiled")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", help="checkpoint folder to merge (default: base model)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    heads = {int(k): v for k, v in yaml.safe_load(open(args.config))["model"]["fixed_heads"].items()}
    model, processor = load_model_and_processor(MODEL, 8, 16, 0.0, [])
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    hooks = AttnHookManager()
    hooks.attach(model, selected_heads_map=heads)
    collate = AttnSupervisionCollator(processor=processor, model_name=MODEL, mask_mode="sam")

    done = sum(1 for _ in open(args.out)) if os.path.exists(args.out) else 0
    with open(args.out, "a") as fout:
        for i, sample in enumerate(sample_tuples(args.n, args.seed)):
            if i < done:
                continue
            batch = collate([sample])
            if not batch.valid_supervision_indices:
                fout.write(json.dumps({"i": i, "skipped": True}) + "\n")
                continue
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.inputs.items()}
            try:
                with torch.no_grad():
                    model(**inputs)
                maps = hooks.attentions
                mask = batch.masks[0].reshape(-1).to(device)
                p, p_t2, rows = [], [], []
                for layer in sorted(heads):
                    attn = maps[layer].float()
                    vis, _ = extract_t2i_attn_valid(attn, batch)
                    p.append(in_mask_mass(vis[0], mask))
                    p_t2.append(code_mass(vis[0], mask))
                    rows.append(extract_t2i_rows(attn, batch)[0][0])
                loss_a3 = a3tune_core_loss([torch.cat(rows, 0)], [batch.image_token_indices[0]], [batch.masks[0]])
            finally:
                hooks.clear()
            p, p_t2 = torch.cat(p, 0), torch.cat(p_t2, 0)                   # heads x tokens
            rec = {"i": i, "mask_frac": float(mask.mean()), "n_img": int(mask.numel()),
                   "n_tokens": int(p.shape[1]), "p_R": p.mean(1).tolist(), "p_code": p_t2.mean(1).tolist(),
                   "p_a3": float(1 - loss_a3.sqrt()) if loss_a3.numel() else None}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(i, f"mask_frac={rec['mask_frac']:.3f} median p_R={float(p.median()):.3f} p_a3={rec['p_a3']}", flush=True)
    summarize(args.out)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if not r.get("skipped")]
    out = {"n_tuples": len(recs), "mask_frac_median": float(np.median([r["mask_frac"] for r in recs])),
           "frac_masks_empty_on_token_grid": float(np.mean([r["mask_frac"] == 0 for r in recs]))}
    for key in ("p_R", "p_code"):
        p = np.concatenate([r[key] for r in recs])                      # one value per (tuple, head)
        push_log, push_sq = 1 - p, 2 * p * (1 - p) ** 2
        out[key] = {
            "quantiles_10_25_50_75_90": np.quantile(p, [.1, .25, .5, .75, .9]).round(4).tolist(),
            "frac_below_0.05": float((p < 0.05).mean()), "frac_below_0.1": float((p < 0.1).mean()),
            "mean_push_neglog": float(push_log.mean()), "mean_push_squared": float(push_sq.mean()),
            "median_ratio_squared_to_neglog": float(np.median(push_sq / np.maximum(push_log, 1e-12))),
            # how the push is spread: share of the total push that goes to the worst-aligned quarter
            "share_of_push_to_lowest_quartile_neglog": float(push_log[p <= np.quantile(p, .25)].sum() / push_log.sum()),
            "share_of_push_to_lowest_quartile_squared": float(push_sq[p <= np.quantile(p, .25)].sum() / push_sq.sum()),
        }
    a3 = np.array([r["p_a3"] for r in recs if r["p_a3"] is not None])
    out["p_a3_quantiles_10_25_50_75_90"] = np.quantile(a3, [.1, .25, .5, .75, .9]).round(4).tolist()
    summary_path = os.path.splitext(path)[0] + "_summary.json"
    json.dump(out, open(summary_path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
