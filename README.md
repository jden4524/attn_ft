# Attention Mask Finetuning

This project fine-tunes selected attention heads in a VLM using segmentation masks aligned to caption phrases.

## Quickstart

1. Create and activate the environment:
   - `python -m venv attn_ft`
   - `source attn_ft/bin/activate`
2. Install dependencies from `pyproject.toml`:
   - `uv pip install -e .`
3. Edit the config as needed:
   - [configs/qwen3_vl_2b_attn_ft.yaml](configs/qwen3_vl_2b_attn_ft.yaml)
4. Launch training:
   - `python -m attn_ft.train --config configs/qwen3_vl_2b_attn_ft.yaml`

## Notes

- The dataset is loaded from [data/flickr30k_sam](data/flickr30k_sam) using the image filename (stem) to locate the folder.
