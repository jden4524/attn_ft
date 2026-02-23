from accelerate import Accelerator
from peft import PeftModel, LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration
from attn_ft.config import load_config
from attn_ft.models import load_model_and_processor


state_fn = "checkpoint-100"  # The specific checkpoint file to load (e.g., "checkpoint-500")


cfg = load_config("configs/qwen3_vl_2b_attn_ft.yaml")

accelerator = Accelerator(
    mixed_precision=cfg.train.mixed_precision,
    gradient_accumulation_steps=cfg.train.grad_accum_steps,
)
torch.manual_seed(cfg.train.seed)

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", 
    torch_dtype="auto", 
    device_map="cpu" # Merging is often safer on CPU to avoid OOM
)
accelerator.load_state(f"outputs/qwen3_vl_2b_attn_ft/{state_fn}")

model, processor = load_model_and_processor(
    cfg.model.name,
    cfg.model.trust_remote_code,
    cfg.model.load_in_4bit,
    cfg.model.lora_r,
    cfg.model.lora_alpha,
    cfg.model.lora_dropout,
    cfg.model.lora_target_modules,
)

model = accelerator.prepare(model)

# 3. Load the checkpoint state
# This loads the LoRA weights into the 'model' object
accelerator.load_state(f"outputs/qwen3_vl_2b_attn_ft/{state_fn}")

# 4. Unwrap and Save the ADAPTER
unwrapped_model = accelerator.unwrap_model(model)

print("Merging LoRA weights into base model...")
merged_model = unwrapped_model.merge_and_unload()

if accelerator.is_main_process:
    merged_model.save_pretrained(
    f"hf_model-2B/{state_fn}",
    state_dict=accelerator.get_state_dict(merged_model) # Crucial for distributed setups
    )
    merged_model.config.save_pretrained(f"hf_model-2B/{state_fn}")
    processor.save_pretrained(f"hf_model-2B/{state_fn}")