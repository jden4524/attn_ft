from __future__ import annotations
import torch.nn.functional as F
import torch


def attn_align_loss(pred: list[torch.Tensor], target: list[torch.Tensor], eps: float = 1.0e-6) -> torch.Tensor:
    losses = []
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            t = t.to(p.device)
            log_probs = F.log_softmax(p, dim=-1)
            pos_loss = -(t * log_probs)
            neg_loss = -(1.0 - t) * torch.log(1.0 - torch.exp(log_probs) + 1e-8)#(1.0 - torch.exp(log_probs)) ** 2 # 
            total_loss = pos_loss + neg_loss
            
            # bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
            losses.append(total_loss.mean())

    return torch.stack(losses).mean()
