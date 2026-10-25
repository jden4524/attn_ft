"""Visual attention redistribution (VAR; Kang et al., ICLR 2025) for the Qwen3-VL language model.

Port of the reference implementation (github.com/seilk/VisAttnSink, src/logic/logic.py):
  1. at prefill, a token is a sink at layer l if |RMSNorm(h)| of its input hidden state
     exceeds tau in one of the sink dimensions;
  2. a (head, query) is image-centric if visual sinks take at most rho of its image
     attention and its total image attention is at least summ;
  3. for image-centric (head, query) pairs, attention to sink tokens is scaled by p and the
     removed mass is given to the non-sink image tokens in proportion to their attention.
Applied from layer 2 to the second-to-last layer, to queries after the image at prefill and
to every generated token. Batch size 1 only. Inactive unless enable() is called.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VARConfig:
    # Qwen3-VL-2B: dim 1793 carries the massive activation of the <|im_start|> sink token.
    # Dim 1999 also exceeds tau on the sink token but on ~99% of image tokens as well, so it
    # marks the image modality rather than sinks and is left out.
    sink_dims: tuple[int, ...] = (1793,)
    tau: float = 20.0
    rho: float = 0.5
    summ: float = 0.2
    p: float = 0.6
    first_layer: int = 2
    skip_last_layer: bool = True


class _State:
    def __init__(self) -> None:
        self.config: VARConfig | None = None
        self.num_layers = 0
        self.image_pos: torch.Tensor | None = None
        self.sinks: dict[int, torch.Tensor] = {}
        self.modified_rows = 0


STATE = _State()


def enable(config: VARConfig | None = None) -> None:
    STATE.config = config or VARConfig()


def disable() -> None:
    STATE.config = None
    STATE.image_pos = None
    STATE.sinks = {}


def is_active() -> bool:
    return STATE.config is not None


def begin_prefill(visual_pos_masks: torch.Tensor | None, num_layers: int) -> None:
    """Reset per-sample state; called once per prefill forward pass."""
    STATE.num_layers = num_layers
    STATE.sinks = {}
    if visual_pos_masks is None:
        STATE.image_pos = None
        return
    if visual_pos_masks.shape[0] != 1:
        raise ValueError("VAR supports batch size 1 only")
    STATE.image_pos = visual_pos_masks[0].nonzero(as_tuple=True)[0]


def record_sinks(hidden_states: torch.Tensor, layer_idx: int) -> None:
    cfg = STATE.config
    if cfg is None or STATE.image_pos is None or layer_idx < cfg.first_layer:
        return
    h = hidden_states[0].float()
    normed = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).abs()
    peak = normed[:, list(cfg.sink_dims)].max(dim=-1).values
    STATE.sinks[layer_idx] = (peak > cfg.tau).nonzero(as_tuple=True)[0]


def redistribute(attn: torch.Tensor, layer_idx: int | None) -> torch.Tensor:
    """attn: post-softmax weights [batch, heads, queries, keys]."""
    cfg = STATE.config
    if cfg is None or layer_idx is None or STATE.image_pos is None:
        return attn
    if layer_idx < cfg.first_layer or (cfg.skip_last_layer and layer_idx == STATE.num_layers - 1):
        return attn
    sinks = STATE.sinks.get(layer_idx)
    if sinks is None or sinks.numel() == 0:
        return attn

    _, _, n_q, n_k = attn.shape
    device = attn.device
    is_img = torch.zeros(n_k, dtype=torch.bool, device=device)
    is_img[STATE.image_pos[STATE.image_pos < n_k].to(device)] = True
    is_sink = torch.zeros(n_k, dtype=torch.bool, device=device)
    is_sink[sinks[sinks < n_k].to(device)] = True
    vis_sink = is_img & is_sink
    if not vis_sink.any():
        return attn

    a = attn.float()
    img_mass = a[..., is_img].sum(-1)
    portion = a[..., vis_sink].sum(-1) / (img_mass + 1e-6)
    image_end = int(STATE.image_pos.max()) + 1
    q_pos = torch.arange(n_k - n_q, n_k, device=device)
    cond = (portion <= cfg.rho) & (img_mass >= cfg.summ) & (q_pos >= image_end)[None, None, :]
    if not cond.any():
        return attn

    non_sink_img = is_img & ~is_sink
    budget = (1.0 - cfg.p) * a[..., is_sink].sum(-1, keepdim=True)
    weights = a[..., non_sink_img]
    ratio = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    new = a.clone()
    new[..., is_sink] *= cfg.p
    new[..., non_sink_img] += budget * ratio
    STATE.modified_rows += int(cond.sum())
    return torch.where(cond[..., None], new, a).to(attn.dtype)
