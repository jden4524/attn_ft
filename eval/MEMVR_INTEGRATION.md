# MemVR Integration for VLMEvalKit

This document describes the integration of MemVR (Memory-augmented Vision Retracing) with Qwen3-VL models in the VLMEvalKit evaluation framework.

## Overview

MemVR enables dynamic vision token retracing in transformer-based vision-language models by:
1. Monitoring entropy of model predictions during generation
2. Triggering vision retracing when entropy exceeds a threshold
3. Injecting vision-informed adapters into mid-layer MLPs
4. Blending adapter output with standard MLP output

This implementation ports MemVR from the M3CoT-specific evaluation in `MemVR/` to the general VLMEvalKit framework in `attn_ft/eval/`, enabling evaluation on multiple benchmarks (POPE, MMVP, MME, HallusionBench, etc.).

## Changes Made

### 1. New Module: `attn_ft/eval/memvr.py`

A self-contained Python module (~280 lines) implementing MemVR for Qwen3-VL models.

**Key functions:**
- `apply_memvr_to_loaded_model(model, ...)` — Main entry point
  - Applies forward hooks and adapters to loaded Qwen3-VL model
  - Configurable: `starting_layer`, `ending_layer`, `entropy_threshold`, `retracing_ratio`
- `_patch_qwen3_vl_memvr()` — Registers all hooks (pre/post/layer-wise)
- `_qwen3_vl_mlp_forward()` — Custom MLP forward with adapter blending
- Entropy monitoring, visual token extraction, adapter initialization

**Rationale:**
- Standalone module avoids cross-repo Git dependencies
- Minimal and simple—suitable for academic prototype
- No changes needed to `MemVR/` directory

### 2. Modified: `attn_ft/eval/VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py`

Extended `Qwen3VLChat` class to support MemVR.

**Changes:**
- Added MemVR imports (with graceful fallback if module unavailable)
- Added init parameters:
  - `apply_memvr: bool = False` — Enable/disable MemVR
  - `memvr_starting_layer: int = 5` — First monitored layer
  - `memvr_ending_layer: int = 16` — Last monitored layer
  - `memvr_entropy_threshold: float = 0.75` — Entropy threshold
  - `memvr_retracing_ratio: float = 0.0` — Adapter blend ratio
- Applied MemVR patches after model loading (non-vllm path only)
- Graceful error handling with logging

**Design decision:** vllm disabled for MemVR models
- vllm uses a separate inference server/engine
- Doesn't expose raw PyTorch model hooks for patching
- Alternative: Use standard transformers inference (slower but reliable)

### 3. Updated: `attn_ft/eval/eval_config.json`

Added MemVR-enabled model configuration.

**New model entry:**
```json
"Qwen3-VL-4B-Instruct-MemVR": {
  "class": "Qwen3VLChat",
  "model_path": "Qwen/Qwen3-VL-4B-Instruct",
  "use_custom_prompt": true,
  "max_new_tokens": 512,
  "use_vllm": false,
  "apply_memvr": true,
  "memvr_starting_layer": 5,
  "memvr_ending_layer": 16,
  "memvr_entropy_threshold": 0.75,
  "memvr_retracing_ratio": 0.0,
  // ... other params ...
}
```

**Baseline preserved:**
- Original `"Qwen3-VL-4B-Instruct"` entry unchanged (with vllm=true)
- Allows side-by-side comparison of MemVR vs. standard inference

## Usage

### Run MemVR-enabled evaluation on a single benchmark:

```bash
cd /home/qqcat/attn_ft/eval

# Using the updated config with MemVR model
python VLMEvalKit/run.py \
  --config eval_config.json \
  --model "Qwen3-VL-4B-Instruct-MemVR" \
  --dataset POPE \
  --work-dir eval_results
```

### Run baseline comparison:

```bash
python VLMEvalKit/run.py \
  --config eval_config.json \
  --model "Qwen3-VL-4B-Instruct" \
  --dataset POPE \
  --work-dir eval_results
```

### Debug MemVR execution:

Enable MemVR debug output (entropy logs, triggered layers):

```bash
export MEMVR_DEBUG=1
python VLMEvalKit/run.py --config eval_config.json --model "Qwen3-VL-4B-Instruct-MemVR" --dataset POPE
```

(Debug output appears as `_memvr_last_state` in model after each forward pass)

## Supported Benchmarks

MemVR-enabled Qwen3-VL can now be evaluated on:
- **VQA**: POPE, MME, HallusionBench, MMVP, VStarBench
- **Vision**: All benchmarks supported by VLMEvalKit

See `data_mapping.json` for full list.

## MemVR Parameters

Default settings (hardcoded in `memvr.py`):
- `starting_layer=5` — Start monitoring entropy from layer 5
- `ending_layer=16` — Stop monitoring at layer 16 (out of 32 layers)
- `entropy_threshold=0.75` — Trigger retracing when normalized entropy > 0.75
- `retracing_ratio=0.0` — 0 = no adapter blending (disabled by default)

To enable vision retracing, set `memvr_retracing_ratio > 0` (e.g., 0.5 for 50% blend).

To override defaults in eval_config.json:
```json
"memvr_retracing_ratio": 0.5,
"memvr_entropy_threshold": 0.8,
```

## Implementation Details

### How MemVR works:

1. **Image features captured** → Stored in first layer MLP
2. **Layer-wise entropy monitoring** → Computed on top-10 logits
3. **Entropy-triggered retracing** → When entropy exceeds threshold:
   - Compute visual-informed adapter weights
   - Inject into next layer's MLP
   - Blend with original MLP output
4. **State reset** → After each forward pass

### vllm Compatibility

**Current status:** vllm disabled for MemVR models

**Reason:** MemVR requires direct access to model layer hooks. vllm uses a separate inference engine (pydantic/async server) that doesn't expose PyTorch hooks.

**Options for future improvement:**
1. Patch vllm's model loading to apply MemVR before engine init
2. Implement MemVR as vllm plugin
3. Use standard transformers inference (current approach—slower but reliable)

If you want to experiment with vllm + MemVR, set `use_vllm: true` in eval_config.json and attempt patching—but expect potential issues.

## Validation

Syntax validation completed:
- ✓ `memvr.py` — Valid Python
- ✓ `model.py` — Valid Python
- ✓ `eval_config.json` — Valid JSON

Next: Runtime testing (see Testing in plan.md)

## Minimal Changes Philosophy

This implementation follows the principle of minimal changes:
- No modifications to MemVR/ directory
- No modifications to VLMEvalKit's core infrastructure
- Only extended Qwen3VLChat class (backward compatible)
- Graceful fallback if memvr module unavailable
- All MemVR logic isolated in single module

This makes it easy to maintain and extend in the future.

## References

- **MemVR paper**: https://arxiv.org/abs/... (update with actual arxiv link)
- **MemVR repo**: https://github.com/1zhou-Wang/MemVR
- **VLMEvalKit**: https://github.com/hengzhan/VLMEvalKit

## Future Enhancements

1. **vllm support**: Port MemVR to work with vllm's inference engine
2. **Per-model config**: Store MemVR params per fine-tuned checkpoint
3. **Adapter variants**: Test different adapter initialization schemes
4. **Multi-model support**: Extend to GLM, LLaVA, etc.
5. **Hyperparameter tuning**: Systematic optimization of entropy threshold, layer ranges

## Troubleshooting

**Q: MemVR module not found**
- A: Ensure you're running from `attn_ft/eval/` directory or add it to `PYTHONPATH`

**Q: Model loads very slowly**
- A: vllm is disabled for MemVR (standard inference is slower). Use baseline model for speed tests.

**Q: No MemVR debug output**
- A: Set `export MEMVR_DEBUG=1` before running. Check model's `_memvr_last_state` attribute after generation.

**Q: Hook registration errors**
- A: Ensure model has expected structure (`model.model.language_model.layers[*].mlp`). Validate with newer transformers (≥4.57.2).
