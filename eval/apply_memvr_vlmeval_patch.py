from __future__ import annotations

import argparse
from pathlib import Path


FLASH_ATTN_WHEEL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl"
PINNED_DATASETS = "datasets==4.8.5"
PINNED_TRANSFORMERS = "transformers==4.57.0"


IMPORT_BLOCK_OLD = """from __future__ import annotations
import logging
import os
import warnings

import torch

from vlmeval.smp import get_gpu_memory, listinstr
from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
"""


IMPORT_BLOCK_NEW = """from __future__ import annotations
import logging
import os
import sys
import warnings

import torch

from vlmeval.smp import get_gpu_memory, listinstr
from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin

# MemVR support
_memvr_import_error = None
try:
    from pathlib import Path
    _eval_dir = Path(__file__).resolve().parents[4]
    if str(_eval_dir) not in sys.path:
        sys.path.insert(0, str(_eval_dir))
    from memvr import apply_memvr_to_loaded_model
    _memvr_available = True
except (ImportError, ModuleNotFoundError) as exc:
    _memvr_import_error = exc
    _memvr_available = False
"""


INIT_SIGNATURE_OLD = """    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
"""


INIT_SIGNATURE_NEW = """    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        apply_memvr: bool = False,
        memvr_starting_layer: int = 5,
        memvr_ending_layer: int = 16,
        memvr_entropy_threshold: float = 0.75,
        memvr_retracing_ratio: float = 0.0,
        **kwargs,
    ) -> None:
"""


MODEL_BLOCK_OLD = """            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()
"""


MODEL_BLOCK_NEW = """            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

            # Apply MemVR patches if requested.
            if apply_memvr:
                if _memvr_available:
                    try:
                        apply_memvr_to_loaded_model(
                            self.model,
                            starting_layer=memvr_starting_layer,
                            ending_layer=memvr_ending_layer,
                            entropy_threshold=memvr_entropy_threshold,
                            retracing_ratio=memvr_retracing_ratio,
                        )
                        logging.info(
                            f"MemVR enabled for {model_path} "
                            f"(layers {memvr_starting_layer}-{memvr_ending_layer}, "
                            f"entropy_threshold={memvr_entropy_threshold})"
                        )
                    except Exception as exc:
                        logging.warning(f"Failed to apply MemVR patches: {exc}")
                else:
                    logging.warning(
                        "MemVR requested but memvr module not available. Proceeding without MemVR. "
                        f"Import error: {_memvr_import_error}"
                    )
"""


LEGACY_DUPLICATE_MODEL_BLOCK = """            # Apply MemVR patches if requested
            if apply_memvr:
                if _memvr_available:
                    try:
                        apply_memvr_to_loaded_model(
                            self.model,
                            starting_layer=memvr_starting_layer,
                            ending_layer=memvr_ending_layer,
                            entropy_threshold=memvr_entropy_threshold,
                            retracing_ratio=memvr_retracing_ratio,
                        )
                        logging.info(f"MemVR enabled for {model_path} (layers {memvr_starting_layer}-{memvr_ending_layer}, entropy_threshold={memvr_entropy_threshold})")
                    except Exception as e:
                        logging.warning(f"Failed to apply MemVR patches: {e}")
                else:
                    logging.warning(
                        "MemVR requested but memvr module not available. Proceeding without MemVR. "
                        f"Import error: {_memvr_import_error}"
                    )
"""


def patch_once(content: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in content:
        return content, False
    if old not in content:
        raise RuntimeError(f"Could not find expected upstream block for {label}.")
    return content.replace(old, new, 1), True


def patch_qwen3_model(model_file: Path) -> bool:
    content = model_file.read_text()
    changed = False

    if LEGACY_DUPLICATE_MODEL_BLOCK in content:
        content = content.replace("\n" + LEGACY_DUPLICATE_MODEL_BLOCK, "", 1)
        changed = True

    if (
        "# MemVR support" in content
        and "apply_memvr_to_loaded_model(" in content
        and "memvr_starting_layer: int = 5" in content
        and MODEL_BLOCK_NEW in content
    ):
        if changed:
            model_file.write_text(content)
        return changed

    content, block_changed = patch_once(content, IMPORT_BLOCK_OLD, IMPORT_BLOCK_NEW, "import block")
    changed = changed or block_changed
    content, block_changed = patch_once(content, INIT_SIGNATURE_OLD, INIT_SIGNATURE_NEW, "__init__ signature")
    changed = changed or block_changed
    content, block_changed = patch_once(content, MODEL_BLOCK_OLD, MODEL_BLOCK_NEW, "model loading block")
    changed = changed or block_changed

    if changed:
        model_file.write_text(content)
    return changed


def ensure_memvr_module(eval_dir: Path) -> None:
    memvr_file = eval_dir / "memvr.py"
    if not memvr_file.exists():
        raise FileNotFoundError(f"Missing MemVR module: {memvr_file}")


def patch_requirements(requirements_file: Path) -> bool:
    if not requirements_file.exists():
        raise FileNotFoundError(f"Could not find requirements file: {requirements_file}")

    lines = requirements_file.read_text().splitlines()
    changed = False

    def pin_requirement(name: str, pinned: str) -> None:
        nonlocal changed, lines
        updated_lines = []
        found = False
        for line in lines:
            if line == pinned:
                found = True
                updated_lines.append(line)
                continue
            if line == name or line.startswith(f"{name}=="):
                if not found:
                    updated_lines.append(pinned)
                    found = True
                changed = True
                continue
            updated_lines.append(line)
        if not found:
            updated_lines.append(pinned)
            changed = True
        lines = updated_lines

    pin_requirement("datasets", PINNED_DATASETS)
    pin_requirement("transformers", PINNED_TRANSFORMERS)

    if FLASH_ATTN_WHEEL not in lines:
        try:
            torch_index = lines.index("torch")
            lines.insert(torch_index + 1, FLASH_ATTN_WHEEL)
        except ValueError:
            lines.append(FLASH_ATTN_WHEEL)
        changed = True

    if changed:
        requirements_file.write_text("\n".join(lines) + "\n")
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch a fresh VLMEvalKit checkout with MemVR Qwen3-VL support.")
    parser.add_argument(
        "--vlmeval-root",
        type=Path,
        default=Path(__file__).resolve().parent / "VLMEvalKit",
        help="Path to the cloned VLMEvalKit repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vlmeval_root = args.vlmeval_root.resolve()
    eval_dir = Path(__file__).resolve().parent
    ensure_memvr_module(eval_dir)

    model_file = vlmeval_root / "vlmeval" / "vlm" / "qwen3_vl" / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Could not find Qwen3-VL model adapter: {model_file}")
    requirements_file = vlmeval_root / "requirements.txt"

    model_changed = patch_qwen3_model(model_file)
    requirements_changed = patch_requirements(requirements_file)

    if model_changed:
        print(f"Patched {model_file}")
    else:
        print(f"MemVR patch already present in {model_file}")

    if requirements_changed:
        print(f"Patched {requirements_file}")
    else:
        print(f"Dependency patch already present in {requirements_file}")


if __name__ == "__main__":
    main()