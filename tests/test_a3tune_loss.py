"""CPU tests of the a3tune_core alignment loss (python -m pytest tests/ or python tests/test_a3tune_loss.py)."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attn_ft.attn_hooks import extract_t2i_rows  # noqa: E402
from attn_ft.data import AttnBatch  # noqa: E402
from attn_ft.losses import a3tune_core_loss  # noqa: E402

NEG = torch.finfo(torch.float32).min


def hand_loss(logits, image_idx, mask):
    """Same quantity with plain Python arithmetic: per-head full-row softmax, mean over heads and
    tokens of the visual columns, then (1 - in-mask / visual)^2."""
    heads, tokens, keys = len(logits), len(logits[0]), len(logits[0][0])
    avg = [0.0] * len(image_idx)
    for h in range(heads):
        for t in range(tokens):
            row = logits[h][t]
            top = max(row)
            exps = [math.exp(v - top) for v in row]
            z = sum(exps)
            for j, k in enumerate(image_idx):
                avg[j] += exps[k] / z / (heads * tokens)
    p_in = sum(a for a, m in zip(avg, mask) if m) / sum(avg)
    return (1 - p_in) ** 2


def test_matches_hand_computed_value():
    # keys: 0 text, 1-4 image, 5-6 text; 2 heads x 2 tokens
    logits = [[[0.5, 1.0, -0.3, 2.0, 0.1, 1.5, -1.0], [0.0, 0.2, 0.4, -0.5, 1.1, 0.3, 0.9]],
              [[2.0, -1.0, 0.0, 0.5, 0.5, -0.2, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]
    image_idx, mask = [1, 2, 3, 4], [0, 1, 1, 0]
    got = a3tune_core_loss([torch.tensor(logits)], [torch.tensor(image_idx)], [torch.tensor(mask, dtype=torch.float32)])
    assert got.shape == (1,)
    assert abs(got.item() - hand_loss(logits, image_idx, mask)) < 1e-6


def test_uses_full_row_softmax():
    # with two heads, the text keys change how much each head's visual map weighs in the average
    logits = torch.tensor([[[0.0, 2.0, 0.0, 0.0]], [[0.0, 0.0, 2.0, 0.0]]])
    image_idx, mask = torch.tensor([1, 2]), torch.tensor([1.0, 0.0])
    base = a3tune_core_loss([logits], [image_idx], [mask]).item()
    shifted = logits.clone()
    shifted[0, 0, 3] = 5.0  # a text key of head 0 takes most of its row
    after = a3tune_core_loss([shifted], [image_idx], [mask]).item()
    assert after > base + 0.05
    # a softmax over the visual columns only would ignore the text keys entirely
    vis = torch.softmax(logits[..., image_idx], -1).mean((0, 1))
    vis_shifted = torch.softmax(shifted[..., image_idx], -1).mean((0, 1))
    assert torch.allclose(vis, vis_shifted)
    # and the full-row value equals the hand computation
    assert abs(after - hand_loss(shifted.tolist(), image_idx.tolist(), mask.tolist())) < 1e-6


def test_range_and_finite_gradients():
    torch.manual_seed(0)
    for trial in range(50):
        heads, tokens, keys, n_img = 6, 3, 40, 24
        scale = 1.0 if trial < 25 else 30.0  # include near-one-hot rows
        logits = (torch.randn(heads, tokens, keys) * scale).requires_grad_()
        image_idx = torch.arange(5, 5 + n_img)
        mask = (torch.rand(n_img) < 0.3).float()
        mask[0] = 1.0
        with torch.no_grad():  # causal / padding entries as the model adds them
            logits[:, :, -3:] = NEG
        loss = a3tune_core_loss([logits], [image_idx], [mask])
        assert 0.0 <= loss.item() <= 1.0
        loss.sum().backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[:, :, -3:].abs().max() == 0  # masked keys get no gradient


def test_gradient_raises_in_mask_mass():
    logits = torch.zeros(4, 2, 10, requires_grad=True)
    image_idx, mask = torch.arange(2, 10), torch.tensor([1, 1, 0, 0, 0, 0, 0, 0.0])
    a3tune_core_loss([logits], [image_idx], [mask]).sum().backward()
    in_mask, out_mask = image_idx[mask.bool()], image_idx[~mask.bool()]
    assert (logits.grad[..., in_mask] < 0).all()   # descent raises in-mask logits
    assert (logits.grad[..., out_mask] > 0).all()
    assert logits.grad.abs().sum() > 0


def test_bf16_rows_and_bounds():
    logits = (torch.randn(3, 2, 16) * 4).to(torch.bfloat16).requires_grad_()
    loss = a3tune_core_loss([logits], [torch.arange(4, 12)], [torch.tensor([1, 0, 0, 1, 0, 0, 0, 0.0])])
    assert loss.dtype == torch.float32 and 0 <= loss.item() <= 1
    loss.sum().backward()
    assert torch.isfinite(logits.grad.float()).all()


def test_perfect_and_empty_masks():
    logits = torch.full((2, 1, 6), NEG)
    logits[:, :, 2] = 0.0  # all attention on image token 2 (image keys 1-4)
    image_idx = torch.tensor([1, 2, 3, 4])
    full = a3tune_core_loss([logits], [image_idx], [torch.tensor([0, 1, 0, 0.0])])
    assert full.item() < 1e-12
    empty = a3tune_core_loss([logits], [image_idx], [torch.zeros(4)])
    assert abs(empty.item() - 1.0) < 1e-12
    # mismatched mask size is skipped, as in soft_suppression_loss
    assert a3tune_core_loss([logits], [image_idx], [torch.zeros(5)]).numel() == 0


def test_extract_rows_keeps_full_key_range():
    bsz, heads, seq = 2, 3, 12
    attn = torch.randn(bsz, heads, seq, seq)
    batch = AttnBatch(
        inputs={}, masks=[torch.ones(2, 2), torch.zeros(2, 2)],
        token_spans=[slice(8, 10), None], image_token_indices=[torch.arange(2, 6), torch.arange(2, 6)],
        valid_supervision_indices=[0], answers=["a", "b"],
    )
    rows, image_idx, masks = extract_t2i_rows(attn, batch)
    assert len(rows) == 1 and rows[0].shape == (heads, 2, seq)
    assert torch.equal(rows[0], attn[0][:, 8:10, :])
    assert torch.equal(image_idx[0], torch.arange(2, 6)) and masks[0].shape == (2, 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
