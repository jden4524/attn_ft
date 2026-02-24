from __future__ import annotations
import torch.nn.functional as F
import torch


def ce_loss(pred: list[torch.Tensor], target: list[torch.Tensor], eps: float = 1.0e-6) -> torch.Tensor:
    losses = []
    if len(pred) == 0 or len(target) == 0:
        return 0
    for p, t in zip(pred, target):
        if t is not None and p is not None:
            t = t.to(p.device)
            log_probs = F.log_softmax(p, dim=-1)
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
            # Scale by temperature
            logits = p / tau
            
            log_total_mass = torch.logsumexp(logits, dim=-1)
            t = t.to(logits.device)
            foreground_logits = logits.masked_fill(t == 0, -1e9)
            log_foreground_mass = torch.logsumexp(foreground_logits, dim=-1)
            losses.append((log_total_mass - log_foreground_mass).mean())
    
    if len(losses) == 0:
        return 0
    return torch.stack(losses).mean()
