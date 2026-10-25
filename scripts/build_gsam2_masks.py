"""Regenerate the grounding masks with Grounded-SAM 2 (GroundingDINO boxes -> SAM 2 masks).

Same phrases and rows as the SAM3 datasets; only the segmenter changes. Following
Algorithm 1, the candidate masks of a phrase are merged into one binary mask, and rows
whose merged mask is empty or covers >= max_fg of the image are dropped.

Generate shards (one per GPU), then merge and push:
  python scripts/build_gsam2_masks.py --src Jackie2235/RLAIF-V-SEG --out-dir gsam2/rlaif --shard 0 --num-shards 4
  python scripts/build_gsam2_masks.py --merge gsam2/rlaif --push-to-hub <user>/RLAIF-V-SEG-gsam2
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Image as HFImage
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="source HF dataset id, e.g. Jackie2235/RLAIF-V-SEG")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None, help="process only the first N rows of the shard")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-fg", type=float, default=0.6, help="foreground-ratio filter of Algorithm 1")
    parser.add_argument("--merge", type=Path, help="merge the shards saved under this directory")
    parser.add_argument("--push-to-hub", help="repo id to push the merged dataset to")
    return parser.parse_args()


def detection_prompt(phrase: str) -> str:
    # GroundingDINO expects lower-case text terminated by a period
    return phrase.strip().lower().rstrip(".") + "."


@torch.no_grad()
def segment_batch(images, phrases, det_proc, det_model, seg_proc, seg_model, args, device):
    inputs = det_proc(images=images, text=[detection_prompt(p) for p in phrases], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = det_model(**inputs)
    detections = det_proc.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[im.size[::-1] for im in images],
    )

    masks = []
    for image, det in zip(images, detections):
        boxes = det["boxes"].tolist()
        if not boxes:
            masks.append(None)
            continue
        seg_inputs = seg_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
        seg_out = seg_model(**seg_inputs, multimask_output=False)
        candidates = seg_proc.post_process_masks(seg_out.pred_masks.cpu(), seg_inputs["original_sizes"].cpu())[0]
        merged = candidates.reshape(-1, *candidates.shape[-2:]).any(dim=0).numpy()
        masks.append(merged)
    return masks


def build_shard(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_proc = AutoProcessor.from_pretrained(args.detector)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(device).eval()
    seg_proc = Sam2Processor.from_pretrained(args.segmenter)
    seg_model = Sam2Model.from_pretrained(args.segmenter).to(device).eval()

    dataset = load_dataset(args.src, split="train").shard(args.num_shards, args.shard, contiguous=True)
    if args.max_rows is not None:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    keep, new_masks = [], []
    stats = {"rows": len(dataset), "no_detection": 0, "too_large": 0}
    for start in range(0, len(dataset), args.batch_size):
        rows = dataset[start : start + args.batch_size]
        images = [im.convert("RGB") for im in rows["image"]]
        masks = segment_batch(images, rows["phrase"], det_proc, det_model, seg_proc, seg_model, args, device)
        for offset, mask in enumerate(masks):
            if mask is None or not mask.any():
                stats["no_detection"] += 1
                continue
            if mask.mean() >= args.max_fg:
                stats["too_large"] += 1
                continue
            keep.append(start + offset)
            buf = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buf, format="PNG")
            new_masks.append({"bytes": buf.getvalue(), "path": None})
        if (start // args.batch_size) % 50 == 0:
            print(f"{start + len(images)}/{len(dataset)} kept={len(keep)}", flush=True)

    out = dataset.select(keep).remove_columns("mask").add_column("mask", new_masks).cast_column("mask", HFImage())
    stats["kept"] = len(keep)
    shard_dir = args.out_dir / f"shard_{args.shard:03d}"
    out.save_to_disk(str(shard_dir))
    (shard_dir / "gsam2_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


def merge(args: argparse.Namespace) -> None:
    shard_dirs = sorted(glob.glob(str(args.merge / "shard_*")))
    merged = concatenate_datasets([load_from_disk(d) for d in shard_dirs])
    totals: dict[str, int] = {}
    for d in shard_dirs:
        for k, v in json.loads((Path(d) / "gsam2_stats.json").read_text()).items():
            totals[k] = totals.get(k, 0) + v
    print("merged", len(merged), "rows; stats", json.dumps(totals))
    if args.push_to_hub:
        merged.push_to_hub(args.push_to_hub, private=True)


def main() -> None:
    args = parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.src or not args.out_dir:
            raise SystemExit("--src and --out-dir are required to build a shard")
        build_shard(args)


if __name__ == "__main__":
    main()
