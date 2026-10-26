"""Held-out grounding, knockout and crop test of the fixed heads on V* (Qwen3-VL-2B, optional adapter).

For each V* question (191; GT target boxes from the V* release):
  * attention of the target-name tokens in the question (never supervised) to image tokens:
    in-box share and pointing hit, per candidate head (layers 16-27);
  * answer at the first answer position: each option is scored by the logsumexp of the logits of
    its single-token forms ("A", " A", "(A"; the model mostly answers "(A"), the margin is the
    score of the correct option minus the best other one, and the answer-format mass is the
    softmax probability (whole vocabulary) of all those forms (`format_mass_bare`: bare letters
    only). `scores_bare` keeps the bare-letter logits used for the base-model analysis of session 1;
  * knockout conditions (logits of the blocked keys set to -inf for text queries after the image),
    each with its letter scores and margin, so the margin drop is margin(none) - margin(cond):
      sel_inbox   selected heads, keys inside the GT boxes
      rand_inbox  layer-matched random candidate heads (2 draws), keys inside the GT boxes
      sel_outbox  selected heads, an equally sized random set of image tokens outside the boxes
  * crop hit: a window covering 1/2 or 1/3 of each side of the token grid is placed where the mean
    attention of the selected heads is largest (from the last position, or from the target-name
    tokens); hit = the target box centre is inside the window. `chance` is the hit rate of a
    uniformly placed window.

The adapter, when given, is a checkpoint folder written by train.py (local path, or a path inside
HF_RUNS_REPO such as checkpoints/<run>/step-<n>); it is merged into the base weights. --var applies
VAR (attn_ft.var) to the language model, for the base+VAR comparison.
Writes one JSON line per question so the run can be resumed or read while in progress.

Usage: python eval/vstar_grounding.py --heads heads.json --vstar-dir <vstar_bench> --out out.jsonl
           [--adapter checkpoints/<run>/step-<n>] [--var] [--limit N]
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import attn_ft.modeling_qwen3_vl as mq  # noqa: E402
from attn_ft import var  # noqa: E402
from attn_ft.models import load_model_and_processor  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
LAYERS = list(range(16, 28))
N_HEADS = 16
CROP_SIDES = {"half": 0.5, "third": 1 / 3}

# ---- attention instrumentation ---------------------------------------------------------
CTX = {"record": None, "block": None}  # record: {layer: rows}; block: {layer: (heads, keys, q_start)}
_orig_eager = mq.eager_attention_forward


def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, return_raw_attn_weights=False, **kw):
    layer = getattr(module, "layer_idx", None)
    if layer is None or (CTX["record"] is None and CTX["block"] is None):
        return _orig_eager(module, query, key, value, attention_mask, scaling, dropout, return_raw_attn_weights, **kw)
    key_states = mq.repeat_kv(key, module.num_key_value_groups)
    value_states = mq.repeat_kv(value, module.num_key_value_groups)
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : key_states.shape[-2]]
    if CTX["block"] is not None and layer in CTX["block"]:
        heads, keys, q_start = CTX["block"][layer]
        if len(heads) and len(keys):
            h = torch.tensor(heads, device=logits.device)
            k = torch.tensor(keys, device=logits.device)
            sub = logits[:, h, q_start:, :]
            sub[..., k] = torch.finfo(logits.dtype).min
            logits[:, h, q_start:, :] = sub
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    if var.is_active():
        weights = var.redistribute(weights, layer)
    if CTX["record"] is not None and layer in LAYERS:
        rows = CTX["record"]["rows"]
        CTX["record"][layer] = weights[0][:, rows, :].float().clone()     # heads x rows x keys
    out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    if return_raw_attn_weights:
        return out, weights, logits
    return out, weights


mq.eager_attention_forward = patched_eager


def box_tokens(bboxes, orig_wh, grid_hw):
    """Indices (row-major in the merged token grid) of tokens overlapping any box [x, y, w, h]."""
    gh, gw = grid_hw
    sx, sy = gw / orig_wh[0], gh / orig_wh[1]
    idx = set()
    for x, y, w, h in bboxes:
        c0, c1 = int(math.floor(x * sx)), int(math.ceil((x + w) * sx)) - 1
        r0, r1 = int(math.floor(y * sy)), int(math.ceil((y + h) * sy)) - 1
        for r in range(max(0, r0), min(gh - 1, max(r0, r1)) + 1):
            for c in range(max(0, c0), min(gw - 1, max(c0, c1)) + 1):
                idx.add(r * gw + c)
    return sorted(idx)


def box_centre_token(bbox, orig_wh, grid_hw):
    gh, gw = grid_hw
    x, y, w, h = bbox
    r = min(gh - 1, int((y + h / 2) * gh / orig_wh[1]))
    c = min(gw - 1, int((x + w / 2) * gw / orig_wh[0]))
    return r, c


def best_window(attn_map, grid_hw, side_frac):
    """Top-left (r, c) and size (wh, ww) of the window with the largest summed attention."""
    gh, gw = grid_hw
    wh, ww = max(1, round(gh * side_frac)), max(1, round(gw * side_frac))
    m = attn_map.reshape(1, 1, gh, gw)
    sums = torch.nn.functional.avg_pool2d(m, (wh, ww), stride=1)[0, 0]       # (gh-wh+1) x (gw-ww+1)
    flat = int(sums.argmax())
    return (flat // sums.shape[1], flat % sums.shape[1]), (wh, ww)


def crop_hit(attn_map, grid_hw, centre, side_frac):
    (r0, c0), (wh, ww) = best_window(attn_map, grid_hw, side_frac)
    r, c = centre
    hit = r0 <= r < r0 + wh and c0 <= c < c0 + ww
    gh, gw = grid_hw
    # windows containing the centre / all window positions (uniform placement)
    rows_ok = min(r, gh - wh) - max(0, r - wh + 1) + 1
    cols_ok = min(c, gw - ww) - max(0, c - ww + 1) + 1
    chance = max(0, rows_ok) * max(0, cols_ok) / ((gh - wh + 1) * (gw - ww + 1))
    return float(hit), chance


def find_span(ids, pattern, start):
    for i in range(start, len(ids) - len(pattern) + 1):
        if ids[i:i + len(pattern)] == pattern:
            return list(range(i, i + len(pattern)))
    return []


def resolve_adapter(adapter):
    if os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download

    root = snapshot_download(os.environ["HF_RUNS_REPO"], allow_patterns=[f"{adapter.strip('/')}/*"])
    path = os.path.join(root, adapter)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise SystemExit(f"no adapter at {adapter} in {os.environ['HF_RUNS_REPO']}")
    return path


def load_model(adapter):
    model, _ = load_model_and_processor(MODEL, 8, 16, 0.0, [])
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolve_adapter(adapter)).merge_and_unload()
    return model.to(torch.bfloat16).eval()


def margin(scores, label):
    return scores[label] - max(v for c, v in scores.items() if c != label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="JSON {layer: [heads]} of the selected heads")
    ap.add_argument("--vstar-dir", required=True, help="vstar_bench folder with test_questions.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", help="checkpoint folder (local or inside HF_RUNS_REPO)")
    ap.add_argument("--var", action="store_true", help="apply VAR to the language model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    selected = {int(k): v for k, v in json.load(open(args.heads)).items()}
    rng = random.Random(0)
    rand_sets = []
    for _ in range(2):
        rand_sets.append({l: sorted(rng.sample(range(N_HEADS), len(selected.get(l, [])))) for l in LAYERS})

    model = load_model(args.adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.config.output_attentions = False
    if args.var:
        var.enable()
    # The analysis only touches language-model attention: run the vision tower with fused
    # SDPA attention and encode each image once for all five conditions.
    model.model.visual.config._attn_implementation = "sdpa"
    vision_cache = {}
    encode_images = model.model.get_image_features

    def cached_image_features(pixel_values, image_grid_thw):
        key = (pixel_values.data_ptr(), tuple(pixel_values.shape))
        if key not in vision_cache:
            vision_cache.clear()
            vision_cache[key] = encode_images(pixel_values, image_grid_thw)
        return vision_cache[key]

    model.model.get_image_features = cached_image_features
    processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    tok = processor.tokenizer
    letter_ids = {c: tok.convert_tokens_to_ids(c) for c in "ABCD"}
    option_ids = {}
    for c in "ABCD":
        forms = [tok.encode(v, add_special_tokens=False) for v in (c, " " + c, "(" + c)]
        option_ids[c] = [f[0] for f in forms if len(f) == 1]
    run_info = {"adapter": args.adapter, "var": args.var}

    vstar = args.vstar_dir
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["question_id"] for l in open(args.out)}
    items = [json.loads(l) for l in open(f"{vstar}/test_questions.jsonl")]
    if args.limit:
        items = items[: args.limit]
    with open(args.out, "a") as fout:
        for it in items:
            if it["question_id"] in done:
                continue
            ann = json.load(open(f"{vstar}/{os.path.splitext(it['image'])[0]}.json"))
            image = Image.open(f"{vstar}/{it['image']}").convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": it["text"]}]}]
            inputs = processor(text=[processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)],
                               images=[image], return_tensors="pt").to(device)
            ids = inputs["input_ids"][0].tolist()
            img_pos = [i for i, t in enumerate(ids) if t == model.config.image_token_id]
            img_start, img_end = img_pos[0], img_pos[-1] + 1
            _, hp, wp = inputs["image_grid_thw"][0].tolist()
            grid = (hp // 2, wp // 2)
            assert grid[0] * grid[1] == len(img_pos)
            letters = [c for c in "ABCD" if f"({c})" in it["text"]]

            # target-name tokens (question part), one box per target
            targets = []
            for name, bbox in zip(ann["target_object"], ann["bbox"]):
                span = []
                for variant in (" " + name, name, " " + name.capitalize(), name.capitalize()):
                    span = find_span(ids, tok.encode(variant, add_special_tokens=False), img_end)
                    if span:
                        break
                targets.append({"span": span, "box": box_tokens([bbox], image.size, grid),
                                "centre": box_centre_token(bbox, image.size, grid)})
            all_box = sorted({t for tg in targets for t in tg["box"]})
            outside = [i for i in range(len(img_pos)) if i not in set(all_box)]
            out_region = sorted(rng.sample(outside, min(len(all_box), len(outside))))

            def predict(block=None, record_rows=None):
                CTX["block"] = block
                CTX["record"] = {"rows": record_rows} if record_rows is not None else None
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1].float()
                rec = CTX["record"]
                CTX["block"] = CTX["record"] = None
                scores = {c: float(torch.logsumexp(logits[option_ids[c]], 0)) for c in letters}
                return max(scores, key=scores.get), scores, rec, logits

            rows = sorted({p for tg in targets for p in tg["span"]} | {len(ids) - 1})
            pred, scores, rec, logits = predict(record_rows=rows)
            row_index = {p: i for i, p in enumerate(rows)}
            probs = torch.softmax(logits, -1)
            answer = {
                "scores_bare": {c: float(logits[letter_ids[c]]) for c in letters},
                "format_mass": float(sum(probs[option_ids[c]].sum() for c in letters)),
                "format_mass_bare": float(sum(probs[letter_ids[c]] for c in letters)),
                "top_token": tok.convert_ids_to_tokens(int(logits.argmax())),
                "margin": margin(scores, it["label"]),
            }

            per_head = {}
            sel_maps = {"last": [], "name": [[] for _ in targets]}  # per-head image maps, normalised
            for layer in LAYERS:
                a = rec[layer][:, :, img_start:img_end]                      # heads x rows x Nimg
                shares, hits = [], []
                for ti, tg in enumerate(targets):
                    if not tg["span"] or not tg["box"]:
                        continue
                    r = [row_index[p] for p in tg["span"]]
                    m = a[:, r, :].mean(1)                                     # heads x Nimg
                    shares.append((m[:, tg["box"]].sum(-1) / m.sum(-1).clamp_min(1e-12)))
                    hits.append(torch.tensor([float(int(m[h].argmax()) in set(tg["box"])) for h in range(m.shape[0])]))
                    for h in selected.get(layer, []):
                        sel_maps["name"][ti].append(m[h] / m[h].sum().clamp_min(1e-12))
                last = a[:, row_index[len(ids) - 1], :]
                for h in selected.get(layer, []):
                    sel_maps["last"].append(last[h] / last[h].sum().clamp_min(1e-12))
                per_head[layer] = {
                    "name_inbox": torch.stack(shares).mean(0).tolist() if shares else None,
                    "name_hit": torch.stack(hits).mean(0).tolist() if hits else None,
                    "last_inbox": (last[:, all_box].sum(-1) / last.sum(-1).clamp_min(1e-12)).tolist(),
                }

            # crop windows from the mean map of the selected heads; one entry per target
            crop = {}
            last_map = torch.stack(sel_maps["last"]).mean(0).cpu()
            for size, frac in CROP_SIDES.items():
                for source in ("last", "name"):
                    hits, chances = [], []
                    for ti, tg in enumerate(targets):
                        if source == "name":
                            if not sel_maps["name"][ti]:
                                continue
                            amap = torch.stack(sel_maps["name"][ti]).mean(0).cpu()
                        else:
                            amap = last_map
                        h, ch = crop_hit(amap, grid, tg["centre"], frac)
                        hits.append(h)
                        chances.append(ch)
                    crop[f"{source}_{size}"] = {"hit": sum(hits) / len(hits), "chance": sum(chances) / len(chances)} \
                        if hits else None

            def block_spec(heads_by_layer, key_tokens):
                keys = [img_start + k for k in key_tokens]
                return {l: (heads_by_layer.get(l, []), keys, img_end) for l in LAYERS}

            knock = {}
            for cond, spec in (("sel_inbox", block_spec(selected, all_box)),
                               ("rand_inbox_0", block_spec(rand_sets[0], all_box)),
                               ("rand_inbox_1", block_spec(rand_sets[1], all_box)),
                               ("sel_outbox", block_spec(selected, out_region))):
                kpred, kscores, _, _ = predict(spec)
                knock[cond] = {"pred": kpred, "scores": kscores, "margin": margin(kscores, it["label"])}
            rec_out = {
                "question_id": it["question_id"], "category": it["category"], "label": it["label"],
                **run_info, "pred": pred, "scores": scores, **answer, "knockout": knock, "crop": crop,
                "n_img_tokens": len(img_pos), "n_box_tokens": len(all_box), "grid": list(grid),
                "box_frac": len(all_box) / len(img_pos), "names_found": [bool(t["span"]) for t in targets],
                "per_head": per_head,
            }
            fout.write(json.dumps(rec_out) + "\n")
            fout.flush()
            drops = {k: round(answer["margin"] - v["margin"], 2) for k, v in knock.items()}
            print(it["question_id"], it["label"], pred, f"mass={answer['format_mass']:.3f}", "drop", drops,
                  f"box_frac={rec_out['box_frac']:.4f}", flush=True)


if __name__ == "__main__":
    main()
