from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from attn_ft.attn_hooks import AttnHookManager, extract_t2i_attn
from attn_ft.config import load_config
from attn_ft.data import AttnSupervisionCollator
from attn_ft.losses import *
from attn_ft.models import (
    filter_trainable_parameters,
    load_model_and_processor
)


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
    )
    torch.manual_seed(cfg.train.seed)

    model, processor = load_model_and_processor(
        cfg.model.name,
        cfg.model.trust_remote_code,
        cfg.model.load_in_4bit,
        cfg.model.lora_r,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
        cfg.model.lora_target_modules,
    )

    attn_manager = AttnHookManager()
    attn_manager.attach(model)

    dataset = load_dataset(cfg.dataset.hf_dataset_id, split="train")

    collator = AttnSupervisionCollator(processor=processor)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    trainable_dict = filter_trainable_parameters(model)
    trainable = list(trainable_dict.values())
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in trainable)
        accelerator.print(f"total parameters: {total_params}")
        accelerator.print(f"trainable parameters: {trainable_params}")
    optimizer = AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    max_steps = cfg.train.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=max_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    if cfg.train.loss == "ce":
        attn_align_loss = ce_loss
        print("Using cross-entropy loss for attention alignment")
    elif cfg.train.loss == "vacuum":
        attn_align_loss = vacuum_loss
        print("Using vacuum loss for attention alignment")
    else:
        raise ValueError(f"Unsupported loss type: {cfg.train.loss}")

    model.train()
    step = 0
    progress = tqdm(
        total=max_steps,
        disable=not accelerator.is_main_process,
        desc="train",
    )
    for epoch in range(cfg.train.num_epochs):
        lm_loss_total = 0.0
        attn_loss_total = 0.0
        
        for batch in dataloader:
            if step >= max_steps:
                break

            with accelerator.accumulate(model):
                batch.inputs.to(accelerator.device)
                labels = batch.labels.to(accelerator.device)
                outputs = model(**batch.inputs, labels=labels)
                all_maps = attn_manager.get_attentions()
                attn_map = all_maps[-2]
                phrase_attn = extract_t2i_attn(attn_map, batch, processor)  # list[Tensor(head, #of text tok, L)]

                align_loss = attn_align_loss(phrase_attn, batch.masks)
                align_loss_item = align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss
                # print(f"LM loss={outputs.loss.item():.4f}")
                # print(f"attn_align_loss={align_loss_item:.4f}")
                lm_loss_total += outputs.loss.item()
                attn_loss_total += align_loss_item
                
                loss = align_loss  + 1.0 * outputs.loss

                attn_manager.clear()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.is_main_process and step % cfg.train.log_every == 0:
                        accelerator.print(f"step={step} lm_loss={lm_loss_total/cfg.train.log_every:.4f} attn_loss={attn_loss_total/cfg.train.log_every:.4f}")
                        lm_loss_total = 0.0
                        attn_loss_total = 0.0

                    if accelerator.is_main_process and step % cfg.train.save_every == 0 and step > 0:
                        out_dir = Path(cfg.train.output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        accelerator.save_state(out_dir / f"checkpoint-{step}")

                    step += 1
                    progress.update(1)

        if step >= max_steps:
            break
    progress.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
