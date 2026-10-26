"""Compare vstar_grounding.py outputs across conditions (paired on the V* questions).

Per condition: in-box attention share and pointing accuracy of the selected heads (target-name
tokens; `_own` variants use the heads a run supervises, given with --own-heads, else the common set), knockout margin drops, answer-format mass, crop hit rates. Per pair of conditions:
Wilcoxon signed-rank tests on the per-question values (questions present in both files).

Usage: python eval/vstar_compare.py --heads heads.json --out summary.json \
           base=base.jsonl spmc_s2000=spmc_2000.jsonl shuffled_s2000=shuf_2000.jsonl ...
       [--pairs spmc_s2000:shuffled_s2000 spmc_s2000:autor_s2000]   (default: all pairs)
       [--own-heads a3tune=heads_a3tune.json]   (runs whose name starts with the prefix)
"""
import argparse
import itertools
import json

import numpy as np
from scipy.stats import wilcoxon

LAYERS = list(range(16, 28))
KNOCK = ("sel_inbox", "rand_inbox_0", "rand_inbox_1", "sel_outbox")


def head_means(r, heads):
    inbox, hit = [], []
    for l in LAYERS:
        ph = r["per_head"][str(l)]
        for h in heads.get(l, []):
            if ph["name_inbox"] is not None:
                inbox.append(ph["name_inbox"][h])
                hit.append(ph["name_hit"][h])
    return (np.mean(inbox), np.mean(hit)) if inbox else (np.nan, np.nan)


def per_question(rows, selected, own=None):
    """question_id -> metric dict."""
    out = {}
    for r in rows:
        inbox, hit = head_means(r, selected)
        inbox_own, hit_own = head_means(r, own) if own else (inbox, hit)
        m = {
            "inbox_share": inbox, "pointing_acc": hit,
            "inbox_share_own": inbox_own, "pointing_acc_own": hit_own,
            "correct": float(r["pred"] == r["label"]),
            "format_mass": r["format_mass"],
            "format_mass_bare": r.get("format_mass_bare", np.nan),
            "margin": r["margin"],
            "box_frac": r["box_frac"],
        }
        for k in KNOCK:
            m[f"drop_{k}"] = r["margin"] - r["knockout"][k]["margin"]
        # knockout specificity: in-box drop of the selected heads beyond random heads
        m["drop_sel_minus_rand"] = m["drop_sel_inbox"] - (m["drop_rand_inbox_0"] + m["drop_rand_inbox_1"]) / 2
        for name, c in r["crop"].items():
            if c is not None:
                m[f"crop_{name}"] = c["hit"]
                m[f"crop_{name}_chance"] = c["chance"]
        out[r["question_id"]] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--own-heads", action="append", default=[], help="prefix=heads.json (repeatable)")
    ap.add_argument("runs", nargs="+", help="name=path.jsonl")
    args = ap.parse_args()
    load = lambda path: {int(k): v for k, v in json.load(open(path)).items()}  # noqa: E731
    selected = load(args.heads)
    own = [(spec.split("=", 1)[0], load(spec.split("=", 1)[1])) for spec in args.own_heads]

    data = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        run_heads = next((h for prefix, h in own if name.startswith(prefix)), None)
        data[name] = per_question([json.loads(l) for l in open(path)], selected, run_heads)

    summary = {"conditions": {}, "pairs": {}}
    for name, q in data.items():
        keys = sorted({k for m in q.values() for k in m})
        summary["conditions"][name] = {"n": len(q), **{k: float(np.nanmean([m.get(k, np.nan) for m in q.values()]))
                                                       for k in keys}}

    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else list(itertools.combinations(data, 2))
    for a, b in pairs:
        common = sorted(set(data[a]) & set(data[b]))
        res = {"n": len(common)}
        for k in sorted(set(data[a][common[0]]) & set(data[b][common[0]])):
            if k.endswith("_chance") or k == "box_frac":
                continue
            x = np.array([data[a][q].get(k, np.nan) for q in common])
            y = np.array([data[b][q].get(k, np.nan) for q in common])
            ok = ~(np.isnan(x) | np.isnan(y))
            d = x[ok] - y[ok]
            entry = {"n": int(ok.sum()), "mean_a": float(x[ok].mean()), "mean_b": float(y[ok].mean()),
                     "mean_diff": float(d.mean()), "frac_a_greater": float((d > 0).mean())}
            entry["wilcoxon_p"] = float(wilcoxon(x[ok], y[ok]).pvalue) if np.any(d != 0) else 1.0
            res[k] = entry
        summary["pairs"][f"{a}:{b}"] = res

    json.dump(summary, open(args.out, "w"), indent=1)
    for name, c in summary["conditions"].items():
        print(name, {k: round(v, 4) for k, v in c.items() if k in
                     ("n", "inbox_share", "pointing_acc", "inbox_share_own", "pointing_acc_own", "drop_sel_inbox", "drop_sel_minus_rand", "format_mass",
                      "correct", "crop_name_half", "crop_last_half")})
    for p, res in summary["pairs"].items():
        print(p, {k: (round(v["mean_diff"], 4), f"p={v['wilcoxon_p']:.3g}") for k, v in res.items()
                  if k in ("inbox_share", "drop_sel_inbox", "drop_sel_minus_rand", "format_mass")})


if __name__ == "__main__":
    main()
