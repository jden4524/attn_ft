"""Run VLMEvalKit with this repo's Qwen3-VL modeling (eager attention), optionally with VAR.

Registers `Qwen3VLChatVAR` in `vlmeval.vlm`, then hands the remaining arguments to
VLMEvalKit's run.py unchanged. Use it in an eval config like:

  "model": {"Qwen3-VL-2B_var": {"class": "Qwen3VLChatVAR", "model_path": "<checkpoint>",
                                "var": true, "temperature": 0.0, "max_new_tokens": 1024, ...}}

  python eval/run_vlmeval.py --vlmevalkit VLMEvalKit -- --config eval_config.json

With "var": false the model runs through the same transformers/eager path without VAR,
which is the matching no-VAR baseline (vLLM runs are not numerically comparable).

Needs transformers<5 (the modified modeling targets 4.x); installing VLMEvalKit's
requirements can upgrade it, so re-pin afterwards.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import attn_ft.models  # noqa: F401  (swaps in the modified Qwen3-VL modeling before transformers loads it)
from attn_ft import var as var_lib


def register(vlmevalkit_dir: Path) -> None:
    sys.path.insert(0, str(vlmevalkit_dir))
    import vlmeval.vlm
    from transformers import AutoModelForImageTextToText
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat

    class Qwen3VLChatVAR(Qwen3VLChat):
        def __init__(
            self,
            model_path: str,
            var: bool = True,
            var_sink_dims: tuple[int, ...] = (1793,),
            var_tau: float = 20.0,
            var_rho: float = 0.5,
            var_summ: float = 0.2,
            var_p: float = 0.6,
            **kwargs,
        ) -> None:
            kwargs["use_vllm"] = False
            had_own = "from_pretrained" in AutoModelForImageTextToText.__dict__
            original = AutoModelForImageTextToText.__dict__.get("from_pretrained")
            bound = AutoModelForImageTextToText.from_pretrained

            def eager_from_pretrained(*args, **kw):
                kw["attn_implementation"] = "eager"
                return bound(*args, **kw)

            AutoModelForImageTextToText.from_pretrained = eager_from_pretrained
            try:
                super().__init__(model_path, **kwargs)
            finally:
                if had_own:
                    AutoModelForImageTextToText.from_pretrained = original
                else:
                    del AutoModelForImageTextToText.from_pretrained
            if var:
                var_lib.enable(
                    var_lib.VARConfig(
                        sink_dims=tuple(var_sink_dims), tau=var_tau, rho=var_rho, summ=var_summ, p=var_p
                    )
                )
            else:
                var_lib.disable()

    vlmeval.vlm.Qwen3VLChatVAR = Qwen3VLChatVAR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlmevalkit", type=Path, required=True, help="path to the VLMEvalKit checkout")
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    register(args.vlmevalkit)
    sys.argv = [str(args.vlmevalkit / "run.py"), *rest]
    runpy.run_path(str(args.vlmevalkit / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
