"""
MemVR (Memory-augmented Vision Retracing) implementation for VLMEvalKit models.

This module provides runtime MemVR patching for compatible Hugging Face VLMs,
currently Qwen3-VL and Mllama-based Llama 3.2 Vision models.

Reference: https://github.com/1zhou-Wang/MemVR
"""

import math
from types import MethodType
import torch
import torch.nn.functional as F


# ==================== Helper Functions ====================

def _ensure_memvr_attrs(mlp):
    """Ensure all MemVR attributes exist on an MLP module."""
    if not hasattr(mlp, "apply_memvr"):
        mlp.apply_memvr = False
    if not hasattr(mlp, "visual_token"):
        mlp.visual_token = None
    if not hasattr(mlp, "retracing_ratio"):
        mlp.retracing_ratio = 0.0
    if not hasattr(mlp, "entropy_threshold"):
        mlp.entropy_threshold = 1.0
    if not hasattr(mlp, "starting_layer"):
        mlp.starting_layer = 0
    if not hasattr(mlp, "ending_layer"):
        mlp.ending_layer = 0
    if not hasattr(mlp, "adpt_sign"):
        mlp.adpt_sign = 0
    if not hasattr(mlp, "adpt_w1"):
        mlp.adpt_w1 = None
    if not hasattr(mlp, "adpt_w2"):
        mlp.adpt_w2 = None


def _clear_memvr_adapter(mlp):
    """Clear MemVR adapter weights from an MLP module."""
    mlp.adpt_sign = 0
    mlp.adpt_w1 = None
    mlp.adpt_w2 = None


def _qwen3_vl_mlp_forward(self, x):
    """Forward pass for Qwen3-VL MLP with optional MemVR adapter."""
    ffn_out = self._memvr_original_forward(x)
    if getattr(self, "adpt_sign", 0) != 1:
        return ffn_out

    adpt_w1 = getattr(self, "adpt_w1", None)
    adpt_w2 = getattr(self, "adpt_w2", None)
    if adpt_w1 is None or adpt_w2 is None:
        return ffn_out

    adapter_out = torch.matmul(torch.matmul(x, adpt_w1.T), adpt_w2.T)
    adapter_scale = torch.mean(torch.abs(ffn_out)) / torch.mean(torch.abs(adapter_out)).clamp_min(1e-6)
    norm_adapter_out = adapter_scale * adapter_out
    retracing_ratio = float(getattr(self, "retracing_ratio", 0.0))
    return (ffn_out * (1 - retracing_ratio)) + (norm_adapter_out * retracing_ratio)


def _flatten_visual_token(visual_token):
    """Normalize captured visual tokens to a 2D [tokens, hidden] tensor."""
    if visual_token is None:
        return None
    if isinstance(visual_token, (list, tuple)):
        if len(visual_token) == 0:
            return None
        visual_token = torch.cat(list(visual_token), dim=0)
    if visual_token.ndim == 3:
        return visual_token.reshape(-1, visual_token.shape[-1])
    return visual_token


def _set_visual_token_on_text_model(text_model, visual_token):
    """Set visual token to the first layer MLP of a decoder-only text backbone."""
    visual_token = _flatten_visual_token(visual_token)
    if visual_token is None or len(text_model.layers) == 0:
        return
    text_model.layers[0].mlp.visual_token = visual_token


def _set_qwen3_vl_visual_token(qwen3_model, visual_token):
    """Set visual token to the first layer MLP of Qwen3-VL language model."""
    _set_visual_token_on_text_model(qwen3_model.language_model, visual_token)


def _extract_qwen3_vl_visual_token(outputs):
    """Extract visual tokens from model.get_image_features() outputs."""
    if outputs is None:
        return None

    if isinstance(outputs, tuple):
        image_embeds = outputs[0]
    else:
        image_embeds = outputs

    return _flatten_visual_token(image_embeds)


def _extract_mllama_visual_token(cross_attention_states):
    """Extract visual tokens from Mllama cross-attention states."""
    return _flatten_visual_token(cross_attention_states)


def _wrap_qwen3_vl_get_image_features(qwen3_model):
    """Wrap Qwen3-VL's get_image_features to capture and store visual tokens."""
    if hasattr(qwen3_model, "_memvr_original_get_image_features"):
        return

    qwen3_model._memvr_original_get_image_features = qwen3_model.get_image_features

    def patched_get_image_features(self, *args, **kwargs):
        outputs = self._memvr_original_get_image_features(*args, **kwargs)
        visual_token = _extract_qwen3_vl_visual_token(outputs)
        _set_qwen3_vl_visual_token(self, visual_token)
        return outputs

    qwen3_model.get_image_features = MethodType(patched_get_image_features, qwen3_model)


def _build_memvr_state(text_model):
    """Build initial MemVR state for a forward pass."""
    layer0_mlp = text_model.layers[0].mlp
    visual_token = getattr(layer0_mlp, "visual_token", None)
    return {
        "active": bool(getattr(layer0_mlp, "apply_memvr", False) and visual_token is not None),
        "visual_token": visual_token,
        "starting_layer": int(getattr(layer0_mlp, "starting_layer", 0)),
        "ending_layer": int(getattr(layer0_mlp, "ending_layer", 0)),
        "entropy_threshold": float(getattr(layer0_mlp, "entropy_threshold", 1.0)),
        "retracing_event": False,
        "pending_reset_layer": None,
        "entropy_list": [],
        "triggered_layers": [],
        "visual_token_shape": tuple(visual_token.shape) if visual_token is not None else None,
    }


def _reset_qwen3_vl_adapters(text_model):
    """Reset all adapters in Qwen3-VL layers."""
    for decoder_layer in text_model.layers:
        _clear_memvr_adapter(decoder_layer.mlp)


def _qwen3_vl_forward_pre_hook(module, args, kwargs):
    """Pre-hook for Qwen3-VL forward: initialize MemVR state."""
    _reset_qwen3_vl_adapters(module)
    module._memvr_state = _build_memvr_state(module)
    return args, kwargs



def _qwen3_vl_forward_post_hook(module, args, kwargs, output):
    """Post-hook for Qwen3-VL forward: clean up MemVR state."""
    state = getattr(module, "_memvr_state", None)
    if state is not None:
        module._memvr_last_state = {
            "active": state["active"],
            "visual_token_shape": state["visual_token_shape"],
            "entropy_count": len(state["entropy_list"]),
            "triggered_layers": list(state["triggered_layers"]),
            "retracing_event": state["retracing_event"],
        }
    _reset_qwen3_vl_adapters(module)
    module._memvr_state = None
    return output


def _make_qwen3_vl_layer_hook(text_model, layer_idx):
    """Create layer-specific hook for MemVR entropy monitoring and adapter triggering."""
    def layer_hook(module, args, kwargs, output):
        state = getattr(text_model, "_memvr_state", None)
        if not state or not state["active"]:
            return output

        hidden_states = output[0] if isinstance(output, tuple) else output
        norm_hidden_states = text_model.norm(hidden_states)
        logits = text_model.lm_head(norm_hidden_states[:, -1, :]).float()

        top_k = min(10, logits.shape[-1])
        top_k_scores = torch.topk(logits, top_k, dim=-1).values
        probabilities = F.softmax(top_k_scores, dim=-1)
        entropy = (-(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=-1) / math.log(top_k)).mean().item()
        state["entropy_list"].append(f"{entropy:.3f}")

        if state["pending_reset_layer"] == layer_idx:
            _clear_memvr_adapter(module.mlp)
            state["pending_reset_layer"] = None

        if (
            entropy > state["entropy_threshold"]
            and not state["retracing_event"]
            and layer_idx > state["starting_layer"]
            and layer_idx < state["ending_layer"]
            and layer_idx + 1 < len(text_model.layers)
            and state["visual_token"] is not None
        ):
            next_mlp = text_model.layers[layer_idx + 1].mlp
            visual_token = state["visual_token"].to(
                device=next_mlp.up_proj.weight.device,
                dtype=next_mlp.up_proj.weight.dtype,
            )
            visual_scale = torch.mean(torch.abs(visual_token)).clamp_min(1e-6)
            next_mlp.adpt_sign = 1
            next_mlp.adpt_w1 = (torch.mean(torch.abs(next_mlp.up_proj.weight)) / visual_scale) * visual_token
            next_mlp.adpt_w2 = (torch.mean(torch.abs(next_mlp.down_proj.weight)) / visual_scale) * visual_token.T

            state["retracing_event"] = True
            state["pending_reset_layer"] = layer_idx + 1
            state["triggered_layers"].append(layer_idx + 1)

        return output

    return layer_hook


def _patch_qwen3_vl_memvr(text_model, qwen3_model):
    """Register all MemVR hooks and patches on Qwen3-VL language model."""
    if hasattr(text_model, "_memvr_handles"):
        return

    _wrap_qwen3_vl_get_image_features(qwen3_model)

    handles = [
        text_model.register_forward_pre_hook(_qwen3_vl_forward_pre_hook, with_kwargs=True),
        text_model.register_forward_hook(_qwen3_vl_forward_post_hook, with_kwargs=True),
    ]

    for layer_idx, decoder_layer in enumerate(text_model.layers):
        _ensure_memvr_attrs(decoder_layer.mlp)
        if not hasattr(decoder_layer.mlp, "_memvr_original_forward"):
            decoder_layer.mlp._memvr_original_forward = decoder_layer.mlp.forward
            decoder_layer.mlp.forward = MethodType(_qwen3_vl_mlp_forward, decoder_layer.mlp)
        handles.append(decoder_layer.register_forward_hook(_make_qwen3_vl_layer_hook(text_model, layer_idx), with_kwargs=True))

    text_model._memvr_handles = handles


def _patch_mllama_memvr(causal_lm, text_backbone):
    """Register all MemVR hooks and patches on Mllama.

    Args:
        causal_lm: MllamaForCausalLM — receives cross_attention_states in forward().
        text_backbone: MllamaTextModel — holds .layers and .norm.
    """
    if hasattr(causal_lm, "_memvr_handles"):
        return

    # Pre-hook on causal_lm: it is the module whose forward() receives cross_attention_states.
    # All layer/state operations are performed on text_backbone (MllamaTextModel).
    def _mllama_pre_hook(module, args, kwargs):
        cross_attention_states = kwargs.get("cross_attention_states", None)
        if cross_attention_states is not None:
            _set_visual_token_on_text_model(text_backbone, _extract_mllama_visual_token(cross_attention_states))
        _reset_qwen3_vl_adapters(text_backbone)
        text_backbone._memvr_state = _build_memvr_state(text_backbone)
        return args, kwargs

    def _mllama_post_hook(module, args, kwargs, output):
        state = getattr(text_backbone, "_memvr_state", None)
        if state is not None:
            text_backbone._memvr_last_state = {
                "active": state["active"],
                "visual_token_shape": state["visual_token_shape"],
                "entropy_count": len(state["entropy_list"]),
                "triggered_layers": list(state["triggered_layers"]),
                "retracing_event": state["retracing_event"],
            }
        _reset_qwen3_vl_adapters(text_backbone)
        text_backbone._memvr_state = None
        return output

    handles = [
        causal_lm.register_forward_pre_hook(_mllama_pre_hook, with_kwargs=True),
        causal_lm.register_forward_hook(_mllama_post_hook, with_kwargs=True),
    ]

    for layer_idx, decoder_layer in enumerate(text_backbone.layers):
        _ensure_memvr_attrs(decoder_layer.mlp)
        if not hasattr(decoder_layer.mlp, "_memvr_original_forward"):
            decoder_layer.mlp._memvr_original_forward = decoder_layer.mlp.forward
            decoder_layer.mlp.forward = MethodType(_qwen3_vl_mlp_forward, decoder_layer.mlp)
        handles.append(decoder_layer.register_forward_hook(_make_qwen3_vl_layer_hook(text_backbone, layer_idx), with_kwargs=True))

    causal_lm._memvr_handles = handles


def _apply_memvr_to_qwen3_vl_model(
    model,
    starting_layer: int,
    ending_layer: int,
    entropy_threshold: float,
    retracing_ratio: float,
):
    if not hasattr(model, "model") or not hasattr(model.model, "language_model"):
        raise TypeError("Expected a Hugging Face Qwen3-VL model with a .model.language_model stack.")

    qwen3_model = model.model
    text_model = qwen3_model.language_model

    if not hasattr(model, "lm_head"):
        raise TypeError("Model is missing 'lm_head' attribute required for MemVR entropy computation.")
    text_model.lm_head = model.lm_head

    _patch_qwen3_vl_memvr(text_model, qwen3_model)

    if len(text_model.layers) == 0:
        raise ValueError("The Qwen3-VL text backbone has no decoder layers to patch.")

    text_model.layers[0].mlp.apply_memvr = True
    text_model.layers[0].mlp.starting_layer = starting_layer
    text_model.layers[0].mlp.ending_layer = ending_layer
    text_model.layers[0].mlp.entropy_threshold = entropy_threshold

    for decoder_layer in text_model.layers:
        decoder_layer.mlp.retracing_ratio = retracing_ratio

    return model


def _apply_memvr_to_mllama_model(
    model,
    starting_layer: int,
    ending_layer: int,
    entropy_threshold: float,
    retracing_ratio: float,
):
    # MllamaForConditionalGeneration
    #   .model  →  MllamaModel
    #     .language_model  →  MllamaForCausalLM   (causal_lm)
    #       .model          →  MllamaTextModel     (text_backbone, has .layers / .norm)
    #       .lm_head        →  Linear
    #     .vision_model    →  MllamaVisionModel
    if not (hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model, "vision_model")):
        raise TypeError("Expected a Hugging Face MllamaForConditionalGeneration with .model.language_model and .model.vision_model.")

    causal_lm = model.model.language_model   # MllamaForCausalLM

    if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
        raise TypeError("MllamaForCausalLM is missing expected .model or .lm_head attributes.")

    text_backbone = causal_lm.model   # MllamaTextModel: has .layers and .norm

    # Attach lm_head to text_backbone so the shared layer hook can compute entropy.
    text_backbone.lm_head = causal_lm.lm_head

    _patch_mllama_memvr(causal_lm, text_backbone)

    if len(text_backbone.layers) == 0:
        raise ValueError("The Mllama text backbone has no decoder layers to patch.")

    text_backbone.layers[0].mlp.apply_memvr = True
    text_backbone.layers[0].mlp.starting_layer = starting_layer
    text_backbone.layers[0].mlp.ending_layer = ending_layer
    text_backbone.layers[0].mlp.entropy_threshold = entropy_threshold

    for decoder_layer in text_backbone.layers:
        decoder_layer.mlp.retracing_ratio = retracing_ratio

    return model


# ==================== Public API ====================

def apply_memvr_to_loaded_model(
    model,
    starting_layer: int = 5,
    ending_layer: int = 16,
    entropy_threshold: float = 0.75,
    retracing_ratio: float = 0.0,
):
    """
    Apply MemVR patches to a loaded compatible Hugging Face VLM.

    Args:
        model: A loaded Qwen3-VL or Mllama conditional generation model instance.
        starting_layer: First layer to monitor for entropy triggering (default: 5).
        ending_layer: Last layer to monitor for entropy triggering (default: 16).
        entropy_threshold: Entropy threshold for triggering vision retracing (default: 0.75).
        retracing_ratio: Blend ratio for retracing adapter output (default: 0.0, i.e., disabled).

    Returns:
        The patched model (same instance, modified in-place).

    Raises:
        TypeError: If model does not have a supported structure.
        ValueError: If language model has no decoder layers.
    """
    cls_name = type(model).__name__

    # Qwen3-VL / Qwen2-VL: outer model has .model.language_model with .layers directly
    if "Qwen" in cls_name:
        return _apply_memvr_to_qwen3_vl_model(
            model,
            starting_layer=starting_layer,
            ending_layer=ending_layer,
            entropy_threshold=entropy_threshold,
            retracing_ratio=retracing_ratio,
        )

    # Llama-3.2-Vision (Mllama): outer model has .model.language_model (MllamaForCausalLM)
    # whose inner .model (MllamaTextModel) holds .layers and .norm
    if cls_name == "MllamaForConditionalGeneration":
        return _apply_memvr_to_mllama_model(
            model,
            starting_layer=starting_layer,
            ending_layer=ending_layer,
            entropy_threshold=entropy_threshold,
            retracing_ratio=retracing_ratio,
        )

    raise TypeError("Expected a supported Hugging Face VLM with a language model stack compatible with MemVR.")
