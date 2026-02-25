from __future__ import annotations
import torch.nn.functional as F
import torch


def ce_loss(pred: list[torch.Tensor], target: list[torch.Tensor], eps: float = 1.0e-6) -> torch.Tensor:
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            t = t.to(p.device).reshape(-1)
            p = p.mean(dim=0).reshape(-1)  # Average over heads and flatten, resulting in shape (H*W)
            log_probs = F.log_softmax(p)
            pos_loss = -(t * log_probs)
            neg_loss = -(1.0 - t) * torch.log(1.0 - torch.exp(log_probs) + 1e-8)#(1.0 - torch.exp(log_probs)) ** 2 # 
            total_loss = pos_loss + neg_loss
            
            # bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
            losses.append(total_loss.mean())
            
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()


def vacuum_loss(pred: list[torch.Tensor], target: list[torch.Tensor], tau=1.0):
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            
            t = t = t.to(p.device).view(-1)
            # Scale by temperature
            rescaled_p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            a = rescaled_p * t
            a_agg = a.mean(dim=-2)

            vacuum_loss = -torch.log((a_agg*t).sum(dim=-1) + 1e-8).mean()
        
            losses.append(vacuum_loss)
    
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()
