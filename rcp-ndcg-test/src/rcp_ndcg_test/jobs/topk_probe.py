#!/usr/bin/env python3
"""topk-embed-v1: the controlled GPU probe for the image-document divergence (investigate lane, Part B).

Runs INSIDE the engine image on the node (the image's python, with the recipe's plugin wheel installed
by the node bootstrap).  The engine side loads vLLM in process and the engine's own HF processor; the
reference side is this same file re-executed under the family's reference venv (sentence-transformers
over the checkpoint's trusted code), which dumps its tensors to a ``.pt`` file.  Every experiment writes
its tensors' shapes, hashes and diffs to ``probe-report.json`` and ``probe-report.md``; nothing is
compared without the numbers being in the report.

    python3 topk_probe.py --out <dir> --model <checkpoint> --revision <rev> --pairs <pairs.jsonl> \
        --reference-python <state>/reference/topk-embed-v1/bin/python [--rows 20,26,27] \
        [--text-rows 17,18] [--query-rows 12] [--dtype bfloat16] [--reference-dtype bfloat16]

    # the reference side (called by the engine side, never by hand):
    <reference-python> topk_probe.py --side reference --image <file.png> --model <checkpoint> \
        --revision <rev> --recipe-dir <recipe dir> --engine-pixels <engine-input.pt> --out <dump.pt>
    <reference-python> topk_probe.py --side reference --text <document or query text> --text-side document \
        --model <checkpoint> --revision <rev> --recipe-dir <recipe dir> --out <dump.pt>

    # the engine image's API contract, seconds and no weights (the wrapper runs it first):
    python3 topk_probe.py --check-engine

Both the engine side and ``--check-engine`` force the single-process engine
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, set before vLLM is imported): the experiments install forward
hooks on the towers and call the vision tower on tensors the probe built, and vLLM's default
multiprocess mode keeps the model in the EngineCore process where neither is reachable.

The experiments (the report's Part B.1 table):
  E0  the control: the probe's engine vectors against the wave's recorded vectors for one row
      (``--control-vectors``; ``not_run`` loudly when absent, never a silent pass).
  E1  preprocessing isolation: the engine's pixels vs the checkpoint's, and every dtype/affine variant
      against BOTH sides (``vs_engine`` is the decisive comparison).
  E2  the same tensors into the engine tower (the engine's pixels and the reference's).
  E3  layer-by-layer inside the vision tower, both feeds, with an in-run bf16 floor.
  E4  the image-token embeddings (the merger output) on the ENGINE's own feed.
  E5  cross-feed: the reference tower on the engine's pixels, and the engine tower on the reference's.
  E6  the grid sweep: the end-to-end per-token cosine for every image document row.
  E7  the text path, layer by layer.  This checkpoint has NO sliding window (``layer_types`` = 18
      ``linear_attention`` + 6 ``full_attention``, ``is_causal: false``), and vLLM builds each decoder
      layer from ``layer_types`` at init, so the brief's window experiment cannot be run here (a
      post-init rewrite is a no-op and would only produce a false exclusion).  E7 compares the text
      model's per-layer hidden states instead; no config is mutated.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

FIRST_DIVERGENCE = 1e-2
"""A tensor's max-abs-diff above this (and above 10x the in-run bf16 floor) names the layer."""

CONTROL_ROW = 26
"""The row whose wave-recorded vectors the control compares against (the worst image row)."""


# ---------------------------------------------------------------------------
# tensor helpers
# ---------------------------------------------------------------------------


def _tensor(value: Any) -> Any:
    """One tensor on the host, float32: detach, cast, move (bf16 has no numpy dtype)."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def _numpy(value: Any) -> Any:
    import numpy as np

    return np.asarray(_tensor(value), dtype=np.float64)


def sha256(value: Any) -> str:
    """The tensor's bytes' hash (dtype and shape included), so equal tensors hash equal."""
    import numpy as np

    array = np.ascontiguousarray(_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:32]


def stats(left: Any, right: Any) -> dict[str, Any]:
    """The comparison of two tensors: shapes, dtypes, hashes, max/mean abs diff and cosine.

    A shape difference that is only singleton dimensions is squeezed for the elementwise comparison and
    recorded in ``note``: vLLM's vision tower keeps a leading 1 on every block's output (its forward
    does ``hidden_states.unsqueeze(1)`` before the blocks) where the checkpoint's returns the plain
    ``[seq, hidden]``, and refusing to compare would read as "clean" -- the same trap the missing
    tensors were.
    """
    import numpy as np

    a = _numpy(left)
    b = _numpy(right)
    out: dict[str, Any] = {
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "left_sha256": sha256(left),
        "right_sha256": sha256(right),
        "equal": bool(a.shape == b.shape and np.array_equal(a, b)),
    }
    if a.shape != b.shape:
        squeezed_a = np.squeeze(a)
        squeezed_b = np.squeeze(b)
        if squeezed_a.shape != squeezed_b.shape:
            out["note"] = "shapes differ: no elementwise comparison"
            return out
        out["note"] = "singleton dimensions squeezed for the comparison (vLLM keeps a leading 1)"
        a, b = squeezed_a, squeezed_b
    diff = np.abs(a - b)
    out["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    out["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    na = float(np.linalg.norm(a.ravel()))
    nb = float(np.linalg.norm(b.ravel()))
    if na and nb:
        out["cosine"] = float(np.dot(a.ravel(), b.ravel()) / (na * nb))
    return out


def bf16_floor(value: Any) -> float:
    """The in-run bf16 rounding floor of one tensor: its max-abs-diff against its own bf16 round trip."""
    import torch

    if value is None:
        return 0.0
    tensor = value.detach() if hasattr(value, "detach") else value
    if not isinstance(tensor, torch.Tensor):
        return 0.0
    return float((tensor.float() - tensor.to(torch.bfloat16).float()).abs().max())


def _record(module: Any, key: str, sink: dict[str, Any]) -> Any:
    """Record one module's forward output in ``sink``; returns the restore callable.

    Deliberately not ``register_forward_hook``: vLLM's ``@support_torch_compile`` decorator replaces a
    decorated class's ``__call__`` with one that calls ``self.forward(...)`` directly
    (``vllm/compilation/decorators.py`` at v0.31.0), so ``nn.Module``'s forward hooks never fire on
    ``Qwen3_VisionPatchEmbed`` and ``Qwen3_VisionBlock`` -- the first node run's E2/E3 tables held the
    undecorated ``merger`` alone, which reads as "the merger is the first divergent layer" when the
    earlier tensors were simply never seen.  Wrapping the bound ``forward`` sees the call either way.
    """
    original = module.forward

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        sink[key] = _tensor(output[0] if isinstance(output, tuple) else output)
        return output

    module.forward = wrapper
    return lambda: setattr(module, "forward", original)


def _record_outputs(module: Any, names: dict[str, Any], sink: dict[str, Any]) -> list[Any]:
    """Wrap ``names``' modules (attribute path -> report key) so each output lands in ``sink``."""
    restores = []
    for path, key in names.items():
        target = module
        for attribute in path.split("."):
            target = getattr(target, attribute)
        restores.append(_record(target, key, sink))
    return restores


def _missing_recorded(names: dict[str, Any], sink: dict[str, Any]) -> list[str]:
    """The keys ``names`` promised that ``sink`` never got: a recording that missed modules says so."""
    return sorted(set(names.values()) - set(sink))


def _vision_hook_names(visual: Any) -> dict[str, Any]:
    return {"patch_embed": "patch_embed", "merger": "merger"} | {
        f"blocks.{index}": f"block_{index}" for index in range(len(visual.blocks))
    }


def _run_recorded(call: Any, module: Any, names: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``call()`` with ``names``' outputs recorded, always restoring the wrapped forwards."""
    sink: dict[str, Any] = {}
    restores = _record_outputs(module, names, sink)
    try:
        output = call()
    finally:
        for restore in restores:
            restore()
    return output, sink


def _text_layer_hook_names(layers: Any) -> dict[str, Any]:
    return {str(index): f"text_layer_{index}" for index in range(len(layers))}


# ---------------------------------------------------------------------------
# the reference side (this file under the reference venv)
# ---------------------------------------------------------------------------


def reference_side(args: argparse.Namespace) -> int:
    """Load the checkpoint's own path, dump its tensors for one image or one text, write ``args.out``."""
    import numpy as np
    import torch
    import torchvision.transforms.v2.functional as tvf
    from PIL import Image
    from sentence_transformers import MultiVectorEncoder

    sys.path.insert(0, str(Path(args.recipe_dir).resolve()))
    import reference as recipe_reference  # the recipe's own transformers shims

    recipe_reference._alias_qwen3_5_layer_type()
    recipe_reference._promote_qwen3_5_rope_position_ids()
    model = MultiVectorEncoder(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        device=args.device,
        model_kwargs={"dtype": args.reference_dtype},
    )
    module = next(m for m in model.modules() if type(m).__name__ == "TopkEmbedTransformer")
    topk = next(m for m in module.modules() if type(m).__name__ == "TopkEmbedModel")
    text_layers = _reference_text_layers(topk)
    if args.text:
        return _reference_text(args, model, module, topk, text_layers)
    visual = topk.model.visual
    device = model.device
    image = Image.open(args.image).convert("RGB")

    # E1: the checkpoint's own row (torchvision antialias BICUBIC + bf16 affine) and its variants.
    row = module._image_row(image)
    target_h, target_w = module._smart_resize(
        image.height,
        image.width,
        factor=module._patch_size * module._merge_size,
        min_pixels=module._min_pixels,
        max_pixels=module._max_pixels,
    )
    torchvision_resized = tvf.resize(
        torch.from_numpy(np.asarray(image)).permute(2, 0, 1),
        [target_h, target_w],
        interpolation=tvf.InterpolationMode.BICUBIC,
        antialias=True,
    )
    pil_resized = torch.from_numpy(np.asarray(image.resize((target_w, target_h), Image.BICUBIC))).permute(2, 0, 1)
    variants = {
        "pixel_values_checkpoint": row.pixel_values,
        "pixel_values_torchvision_fp32_affine": _affine(module, torchvision_resized, dtype=torch.float32),
        "pixel_values_pil_fp32_affine": _affine(module, pil_resized, dtype=torch.float32),
        "pixel_values_pil_bf16_affine": _affine(module, pil_resized, dtype=torch.bfloat16),
    }

    vision_inputs = {
        key.removeprefix("vision_"): value for key, value in module._vision_inputs([row.image_grid_thw]).items()
    }
    vision_inputs = _to_device(vision_inputs, device)
    pixels = row.pixel_values.to(device=device, dtype=visual.patch_embed.proj.weight.dtype)
    with torch.no_grad():
        merged, layers = _run_recorded(
            functools.partial(visual, pixels, **vision_inputs), visual, _vision_hook_names(visual)
        )
    layers["merged"] = _tensor(merged)

    # E5: the reference tower on the ENGINE's pixels (the cross-feed leg).
    engine_layers: dict[str, Any] = {}
    engine_meta: dict[str, Any] = {}
    if args.engine_pixels and Path(args.engine_pixels).is_file():
        engine_input = torch.load(args.engine_pixels, map_location="cpu", weights_only=False)
        engine_pixels = engine_input["pixel_values"].to(device=device, dtype=visual.patch_embed.proj.weight.dtype)
        engine_grid = engine_input["grid_thw"]
        engine_vision_inputs = {
            key.removeprefix("vision_"): value for key, value in module._vision_inputs([engine_grid]).items()
        }
        engine_vision_inputs = _to_device(engine_vision_inputs, device)
        with torch.no_grad():
            engine_merged, engine_layers = _run_recorded(
                functools.partial(visual, engine_pixels, **engine_vision_inputs),
                visual,
                _vision_hook_names(visual),
            )
        engine_layers["merged"] = _tensor(engine_merged)
        engine_meta = {"grid_thw": engine_grid, "shape": list(engine_pixels.shape)}

    # E7: the text path's per-layer hidden states, captured during the card's own encode.
    with torch.no_grad():
        document, text_hooks = _run_recorded(
            functools.partial(model.encode_document, [image]),
            text_layers,
            _text_layer_hook_names(text_layers),
        )
        document = document[0]
    torch.save(
        {
            "grid_thw": row.image_grid_thw,
            "target": (target_h, target_w),
            "layers": layers,
            "variants": {key: _tensor(value) for key, value in variants.items()},
            "document_vectors": _tensor(document),
            "engine_pixels_layers": engine_layers,
            "engine_pixels_meta": engine_meta,
            "text_layers": text_hooks,
        },
        args.out,
    )
    print(f"reference: {args.image} -> {args.out} ({len(layers)} vision tensors, {len(text_hooks)} text layers)")
    return 0


def _reference_text(args: argparse.Namespace, model: Any, module: Any, topk: Any, text_layers: Any) -> int:
    """The reference side for one text row: the card's own render, its vectors and its text layers.

    The checkpoint's keep mask is recorded as ``kept_positions`` so the engine side can select the
    same positions out of its own vectors -- the engine applies the *image* allowlist only (the text
    rule is the client's), and comparing all positions against the reference's kept set would compare
    different token sets.
    """
    import torch

    features = module.preprocess([args.text], task=args.text_side)
    kept = features["scoring_mask"][0].bool().nonzero().squeeze(1).tolist()
    encode = model.encode_query if args.text_side == "query" else model.encode_document
    with torch.no_grad():
        document, text_hooks = _run_recorded(
            functools.partial(encode, [args.text]),
            text_layers,
            _text_layer_hook_names(text_layers),
        )
    torch.save(
        {
            "kind": "text",
            "side": args.text_side,
            "document_vectors": _tensor(document[0]),
            "text_layers": text_hooks,
            "kept_positions": kept,
        },
        args.out,
    )
    print(f"reference: {args.text_side} ({len(args.text)} chars) -> {args.out} ({len(text_hooks)} text layers)")
    return 0


def _to_device(mapping: dict[str, Any], device: Any) -> dict[str, Any]:
    """One vision-inputs mapping with every tensor on the model's device."""
    import torch

    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in mapping.items()}


def _reference_text_layers(topk: Any) -> Any:
    """The checkpoint text model's decoder layers (the packed Qwen3.5 text model)."""
    for name, module in topk.named_modules():
        if name.endswith("language_model") and hasattr(module, "layers"):
            return module.layers
    raise RuntimeError("cannot reach the checkpoint's text layers")


def _affine(module: Any, image: Any, *, dtype: Any) -> Any:
    """The checkpoint's ``_patchify`` affine on one resized frame, at ``dtype`` (the dtype variant)."""
    previous = module.pixel_dtype
    module.pixel_dtype = dtype
    try:
        return module._patchify(image)
    finally:
        module.pixel_dtype = previous


# ---------------------------------------------------------------------------
# the engine side
# ---------------------------------------------------------------------------

#: The in-process route to the loaded model, in the order v0.31.0 exposes it.  ``llm.llm_engine`` is an
#: ``LLMEngine``; its ``model_executor`` exists only when the EngineCore runs in this process, which
#: ``force_single_process_engine`` forces (v0.31.0's ``LLMEngine.__init__`` sets it in its
#: ``not multiprocess_mode`` branch, and its own ``_get_driver_model_for_cleanup`` reads
#: ``driver_worker.model_runner.model``).  The second path is the worker-wrapped executor shape, the
#: third the same route without v0.31.0's v0-compatibility alias.
ENGINE_MODEL_PATHS: tuple[tuple[str, ...], ...] = (
    ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
    ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
    ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
)


def force_single_process_engine() -> None:
    """Run vLLM's EngineCore in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``).

    The probe installs forward hooks on the towers and calls the vision tower on tensors it built
    itself; in vLLM's default multiprocess mode the model lives in the EngineCore process and neither
    is reachable here (v0.31.0's ``uniproc_executor.py`` names this switch as the way to run the
    engine in one process).  Must run before vLLM is imported: vLLM reads the switch when it builds
    the engine.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def engine_model(llm: Any) -> Any:
    """The loaded vLLM model object, through :data:`ENGINE_MODEL_PATHS`; a typed error if it moved."""
    tried: list[str] = []
    for path in ENGINE_MODEL_PATHS:
        target: Any = llm
        for attribute in path:
            target = getattr(target, attribute, None)
            if target is None:
                break
        tried.append(".".join(path))
        if target is not None:
            return target
    raise RuntimeError(
        "cannot reach the engine's model object: tried "
        + ", ".join(tried)
        + ".  The in-process model is only reachable when the EngineCore runs in this process "
        "(VLLM_ENABLE_V1_MULTIPROCESSING=0, set by force_single_process_engine before vLLM is "
        "imported); if the attribute path moved, re-read vllm/v1/engine/llm_engine.py's "
        "non-multiprocess branch and ENGINE_MODEL_PATHS against the engine image."
    )


def engine_visual(model: Any) -> Any:
    """The engine's vision tower (``visual`` on the model or on its inner model); loud if moved."""
    visual = getattr(model, "visual", None)
    if visual is None:
        visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise RuntimeError(
            f"cannot reach the engine's vision tower: tried model.visual and model.model.visual on "
            f"{type(model).__name__} (the plugin's ColQwen3_5Model subclass exposes self.visual)"
        )
    return visual


def engine_text_layers(model: Any) -> Any:
    """The engine text model's decoder layers; loud if the model's shape moved."""
    for name, module in model.named_modules():
        if "language_model" in name and hasattr(module, "layers"):
            return module.layers
    raise RuntimeError(
        f"cannot reach the engine's text layers on {type(model).__name__}: no module under "
        "language_model carries .layers (the plugin's model nests the decoder at "
        "language_model.model.layers)"
    )


def engine_side(args: argparse.Namespace) -> int:
    """Load vLLM in process, run E0-E7 for the requested rows, write the report."""
    force_single_process_engine()  # before the vLLM import: the model must be reachable in this process
    import torch
    from transformers import AutoProcessor
    from vllm import LLM

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "engine_image": os.environ.get("RCP_IMAGE", "unknown"),
        "dtype": args.dtype,
        "reference_dtype": args.reference_dtype,
        "device": args.device,
        "reference_device": args.reference_device,
        "experiments": {},
        "rows": [],
    }
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    image_processor = processor.image_processor
    report["experiments"]["E1_preprocessing"] = {
        "hypothesis": "the engine's PIL BICUBIC + fp32 affine pixels differ from the checkpoint's "
        "torchvision antialias BICUBIC + bf16 affine pixels",
        "engine_processor": type(image_processor).__name__,
        "engine_call": "the HF image processor at the pinned budget: the same call vLLM's overlay makes "
        "from the recipe's nested images_kwargs",
        "pin": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        "comparison": "each variant is compared against BOTH the engine's pixels (vs_engine, the decisive "
        "one) and the checkpoint's (vs_checkpoint)",
    }
    report["experiments"]["E7_text_layers"] = {
        "note": "this checkpoint has no sliding window (layer_types = 18 linear_attention + 6 "
        "full_attention, is_causal false) and vLLM builds the decoder layers from layer_types at init, "
        "so the brief's window experiment is invalid here; the text path's per-layer states are compared "
        "instead and no config is mutated",
    }

    llm = LLM(
        model=args.model,
        revision=args.revision,
        runner="pooling",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_fraction,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"images_kwargs": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}},
        hf_overrides={"document_keep_token_ids": [args.image_token_id]},
    )
    model = engine_model(llm)
    visual = engine_visual(model)
    text_layers = engine_text_layers(model)
    report["experiments"]["E1_preprocessing"]["engine_visual"] = type(visual).__name__

    rows = _pairs_rows(args)
    control = _read_control(args)
    for row_index, document_index, image in rows:
        row_report: dict[str, Any] = {
            "pairs_index": row_index,
            "document_index": document_index,
            "image": str(image.get("sha256", ""))[:16],
        }
        try:
            pil = _load_image(image)
            row_report["width"], row_report["height"] = pil.width, pil.height
            # E1: the engine's own preprocessing.
            processed = image_processor(
                images=[pil], return_tensors="pt", min_pixels=args.min_pixels, max_pixels=args.max_pixels
            )
            engine_pixels = processed["pixel_values"]
            engine_grid = processed["image_grid_thw"]
            row_report["engine_grid_thw"] = engine_grid.tolist()
            row_report["engine_pixel_values"] = {
                "shape": list(engine_pixels.shape),
                "dtype": str(engine_pixels.dtype),
                "sha256": sha256(engine_pixels),
            }
            # the reference side, in its own venv (and on its own GPU).
            image_path = out_dir / f"row{row_index}-doc{document_index}.png"
            pil.save(image_path)
            engine_input_path = out_dir / f"row{row_index}-doc{document_index}-engine-input.pt"
            torch.save({"pixel_values": _tensor(engine_pixels), "grid_thw": engine_grid}, engine_input_path)
            dump_path = out_dir / f"row{row_index}-doc{document_index}-reference.pt"
            subprocess.run(_reference_command(args, image_path, engine_input_path, dump_path), check=True)
            reference = torch.load(dump_path, map_location="cpu", weights_only=False)
            row_report["reference_grid_thw"] = reference["grid_thw"].tolist()
            row_report["reference_target_hw"] = list(reference["target"])
            row_report["grids_agree"] = reference["grid_thw"].tolist() == engine_grid.tolist()
            # E1: the pixel tensors, every variant against BOTH sides.
            row_report["E1_pixel_values"] = stats(engine_pixels, reference["variants"]["pixel_values_checkpoint"])
            row_report["E1_variants"] = {
                name: {
                    "shape": list(_numpy(value).shape),
                    "sha256": sha256(value),
                    "vs_engine": stats(engine_pixels, value),
                    "vs_checkpoint": stats(value, reference["variants"]["pixel_values_checkpoint"]),
                }
                for name, value in reference["variants"].items()
            }
            # E2/E3: the engine tower on both feeds, with the in-run bf16 floor.
            vision_names = _vision_hook_names(visual)
            engine_on_engine, dumps_engine = _run_recorded(
                functools.partial(
                    visual,
                    engine_pixels.to(device=args.device, dtype=visual.patch_embed.proj.weight.dtype),
                    engine_grid.to(args.device),
                ),
                visual,
                vision_names,
            )
            engine_on_reference, dumps_reference = _run_recorded(
                functools.partial(
                    visual,
                    reference["variants"]["pixel_values_checkpoint"].to(
                        device=args.device, dtype=visual.patch_embed.proj.weight.dtype
                    ),
                    torch.as_tensor(reference["grid_thw"]).to(args.device),
                ),
                visual,
                vision_names,
            )
            dumps_engine["merged"] = _tensor(engine_on_engine)
            dumps_reference["merged"] = _tensor(engine_on_reference)
            # A recording that missed modules must say so: an absent tensor reads as "clean" otherwise.
            missing = _missing_recorded(vision_names, dumps_engine)
            if missing:
                row_report["E3_missing_tensors"] = missing
            row_report["bf16_floor"] = max(
                bf16_floor(engine_pixels),
                bf16_floor(dumps_engine.get("patch_embed")),
            )
            row_report["E2_tower_on_engine_pixels"] = {
                key: stats(value, reference["layers"].get(key)) for key, value in dumps_engine.items()
            }
            row_report["E2_tower_on_reference_pixels"] = {
                key: stats(value, reference["layers"].get(key)) for key, value in dumps_reference.items()
            }
            # E4: the image-token embeddings on the ENGINE's own feed.
            row_report["E4_image_token_embeddings"] = stats(dumps_engine["merged"], reference["layers"]["merged"])
            # E5: the cross-feed, both directions.
            row_report["E5_cross_feed"] = {
                "engine_tower_on_reference_pixels_vs_own": stats(dumps_reference["merged"], dumps_engine["merged"]),
                "reference_tower_on_engine_pixels_vs_own": stats(
                    reference["engine_pixels_layers"].get("merged"), reference["layers"]["merged"]
                ),
                "reference_tower_on_engine_pixels_vs_engine_tower": stats(
                    reference["engine_pixels_layers"].get("merged"), dumps_engine["merged"]
                ),
                "engine_pixels_shape": reference["engine_pixels_meta"].get("shape"),
            }
            # E6: the engine's end-to-end token vectors for the same image document, and the reference's.
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            placeholders = rendered.count(processor.image_token)
            row_report["render"] = {
                "source": "the checkpoint's own image-only chat render (the recipe's image document render)",
                "image_placeholders": placeholders,
            }
            engine_vectors = None
            if placeholders != 1:
                row_report["E6_end_to_end"] = {
                    "note": "the row's image is not the document's only part: the image-only render does "
                    "not apply (the recipe's reference refuses a mixed document)",
                    "mismatch": True,
                }
            else:
                engine_vectors, engine_text_hooks = _run_recorded(
                    functools.partial(_engine_encode, llm, rendered, pil),
                    text_layers,
                    _text_layer_hook_names(text_layers),
                )
                reference_vectors = reference["document_vectors"]
                cosine = _per_token_cosine(engine_vectors, reference_vectors)
                cosine["wave_min_cosine"] = (args.wave_cosines or {}).get(str(row_index))
                row_report["E6_end_to_end"] = {
                    "engine_tokens": int(len(engine_vectors)),
                    "reference_tokens": int(len(reference_vectors)),
                    "cosine_per_token": cosine,
                }
                # E7: the text path, layer by layer (engine hooks vs the reference's dump).
                row_report["E7_text_layers"] = {
                    key: stats(value, reference["text_layers"].get(key)) for key, value in engine_text_hooks.items()
                }
            # E0: the control, when the wave's recorded vectors for the control row are provided.
            if control is not None and row_index == control["row"] and engine_vectors is not None:
                row_report["E0_control"] = {
                    "row": control["row"],
                    "wave_vectors": int(len(control["vectors"])),
                    "probe_vectors": int(len(engine_vectors)),
                    "cosine_per_token": _per_token_cosine(engine_vectors, control["vectors"]),
                }
            row_report["passed"] = True
        except Exception as error:  # noqa: BLE001 - one failed row must not stop the sweep
            row_report["passed"] = False
            row_report["error"] = f"{type(error).__name__}: {error}"
            row_report["traceback"] = traceback.format_exc()
        report["rows"].append(row_report)
        _write_report(out_dir, report)

    # The text-only rows (their own loop: no vision tower, no pixels, no render question).
    for row_index, item in _text_rows(args):
        row_report = {
            "pairs_index": row_index,
            "document_index": None,
            "kind": "text",
            "side": item["side"],
            "chars": len(item["text"]),
        }
        try:
            dump_path = out_dir / f"row{row_index}-{item['side']}-reference.pt"
            subprocess.run(_reference_command(args, None, None, dump_path, item), check=True)
            reference = torch.load(dump_path, map_location="cpu", weights_only=False)
            prompt = _text_render(args, item)
            row_report["render"] = {
                "source": "the checkpoint's own render (format_document / format_query, the recipe's client template)",
                "chars": len(prompt),
            }
            engine_vectors, engine_text_hooks = _run_recorded(
                functools.partial(_engine_encode, llm, prompt),
                text_layers,
                _text_layer_hook_names(text_layers),
            )
            kept = torch.as_tensor(reference["kept_positions"], dtype=torch.long)
            engine_kept = engine_vectors[kept.numpy()] if kept.numel() else engine_vectors
            cosine = _per_token_cosine(engine_kept, reference["document_vectors"])
            cosine["wave_min_cosine"] = (args.wave_cosines or {}).get(str(row_index))
            row_report["E6_end_to_end"] = {
                "engine_tokens": int(len(engine_kept)),
                "reference_tokens": int(len(reference["document_vectors"])),
                "cosine_per_token": cosine,
            }
            row_report["E7_text_layers"] = {
                key: stats(value, reference["text_layers"].get(key)) for key, value in engine_text_hooks.items()
            }
            row_report["passed"] = True
        except Exception as error:  # noqa: BLE001 - one failed row must not stop the sweep
            row_report["passed"] = False
            row_report["error"] = f"{type(error).__name__}: {error}"
            row_report["traceback"] = traceback.format_exc()
        report["rows"].append(row_report)
        _write_report(out_dir, report)

    report["control"] = _control_summary(report, control)
    report["first_divergent_layer"] = _first_divergent(report, "E2_tower_on_reference_pixels")
    report["first_divergent_layer_engine_pixels"] = _first_divergent(report, "E2_tower_on_engine_pixels")
    _write_report(out_dir, report)
    return 0


def check_engine() -> int:
    """The engine image's API contract, checked in seconds without loading a model or a GPU.

    The wrapper runs this on the node before the probe, so a moved vLLM API fails the job before the
    weights load: the switch that puts the EngineCore in this process is honoured, ``LLMEngine``'s
    non-multiprocess branch still exposes the ``model_executor`` the probe walks to, and the offline
    pooling API E6 calls is present.  Prints the facts as JSON; exit 0 only when every one holds.
    """
    force_single_process_engine()
    import inspect

    try:
        import vllm
        from vllm import LLM
        from vllm.v1.engine.llm_engine import LLMEngine
    except Exception as error:  # noqa: BLE001 - the message is the artefact
        print(f"check-engine: cannot import vLLM in this image: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    facts: dict[str, Any] = {
        "vllm": getattr(vllm, "__version__", "unknown"),
        "single_process_switch": "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "engine_model_paths": [".".join(path) for path in ENGINE_MODEL_PATHS],
    }
    problems: list[str] = []
    if getattr(vllm.envs, "VLLM_ENABLE_V1_MULTIPROCESSING", None) is not False:
        problems.append("vllm.envs does not read VLLM_ENABLE_V1_MULTIPROCESSING as False after it is set")
    # The line the first ENGINE_MODEL_PATHS entry rides on, read from the class the image ships.
    if "self.engine_core.engine_core.model_executor" not in inspect.getsource(LLMEngine.__init__):
        problems.append(
            "LLMEngine.__init__ no longer assigns model_executor from the in-process engine core "
            "(re-read ENGINE_MODEL_PATHS against vllm/v1/engine/llm_engine.py)"
        )
    for name in ("encode", "apply_model"):
        if not hasattr(LLM, name):
            problems.append(f"LLM.{name} is gone (the offline pooling API moved)")
    facts["problems"] = problems
    print(json.dumps(facts, indent=1))
    if problems:
        print("check-engine: the engine image's vLLM API moved:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


def _reference_command(
    args: argparse.Namespace,
    image_path: Any,
    engine_input_path: Any,
    dump_path: Any,
    item: dict[str, Any] | None = None,
) -> list[str]:
    """The reference subprocess' argv: the same module, its own device and its own load dtype.

    ``--reference-dtype`` must travel here: the reference side loads the checkpoint itself, so a
    dtype experiment that moves only the engine compares two precisions (the first fp32 run did, and
    its rows all read the engine's fp32 tower against the reference's bf16 one).  A text row carries
    its own text and side instead of an image (no vision path runs for it).
    """
    command = [
        args.reference_python,
        str(Path(__file__).resolve()),
        "--side",
        "reference",
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--recipe-dir",
        args.recipe_dir,
        "--device",
        args.reference_device,
        "--reference-dtype",
        args.reference_dtype,
        "--out",
        str(dump_path),
    ]
    if item is not None:
        command += ["--text", item["text"], "--text-side", item["side"]]
    else:
        command += ["--image", str(image_path), "--engine-pixels", str(engine_input_path)]
    return command


def _read_control(args: argparse.Namespace) -> dict[str, Any] | None:
    """The wave-recorded vectors for the control row (``--control-vectors``), or None."""
    if not args.control_vectors:
        return None
    path = Path(args.control_vectors)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"row": int(data.get("row", CONTROL_ROW)), "vectors": data["vectors"]}


def _control_summary(report: dict[str, Any], control: dict[str, Any] | None) -> Any:
    """The control's status: run with its numbers, or ``not_run`` with the reason (never a silent pass)."""
    if control is None:
        return {
            "status": "not_run",
            "reason": "no --control-vectors file: the probe's engine vectors are not tied to the wave's "
            "recorded vectors, so every E-number is uncalibrated",
            "how": 'dump one wave-recorded row\'s per-token vectors to {"row": <row>, "vectors": [[...]]} '
            "and pass it (the observation corpus' response for request_id pairs:<row>)",
        }
    for row in report["rows"]:
        if row.get("pairs_index") == control["row"] and "E0_control" in row:
            return {"status": "run", **row["E0_control"]}
    return {"status": "not_run", "reason": f"row {control['row']} was not probed with an end-to-end encode"}


def _engine_encode(llm: Any, prompt: str, image: Any = None) -> Any:
    """One document through the engine's own encode (the wave's signal, in process); no image = text."""
    request: dict[str, Any] = {"prompt": prompt}
    if image is not None:
        request["multi_modal_data"] = {"image": image}
    outputs = llm.encode([request], pooling_task="token_embed", use_tqdm=False)
    return _numpy(outputs[0].outputs.data)


def _per_token_cosine(engine_vectors: Any, reference_vectors: Any) -> dict[str, Any]:
    """The per-token cosine between the engine's kept vectors and the reference's kept vectors."""
    import numpy as np

    engine = np.asarray(engine_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if engine.ndim == 1:
        engine = engine.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    if engine.shape != reference.shape:
        return {
            "mismatch": True,
            "note": "token counts differ: no cosine",
            "engine_shape": list(engine.shape),
            "reference_shape": list(reference.shape),
        }
    na = np.linalg.norm(engine, axis=1)
    nb = np.linalg.norm(reference, axis=1)
    cosines = (engine * reference).sum(axis=1) / np.maximum(na * nb, 1e-12)
    return {
        "min": float(cosines.min()),
        "mean": float(cosines.mean()),
        "below_0_9": int((cosines < 0.9).sum()),
        "n": int(cosines.size),
    }


def _index_set(value: Any) -> set[int]:
    """A list of pairs indices (``--rows`` / ``--text-rows`` / ``--query-rows``) as a set.

    Commas, semicolons and plus signs all separate: the job CLI renders an operator's knob through
    helm's ``--set``, which splits on commas (``env.RCP_TOPK_TEXT_ROWS=17,18`` is a parse error), so
    a node submission passes the list with ``;`` while a hand-run argv may use commas.
    """
    return {int(item) for item in re.split(r"[,;+]", str(value)) if item.strip()}


def _pairs_rows(args: argparse.Namespace) -> list[tuple[int, int, dict[str, Any]]]:
    """The pairs rows carrying an image document (or the ``--rows`` selection): (row, document, entry)."""
    rows: list[tuple[int, int, dict[str, Any]]] = []
    selected = _index_set(args.rows)
    for index, row in enumerate(_read_pairs(args.pairs)):
        if selected and index not in selected:
            continue
        for position, entry in enumerate((row.get("media") or {}).get("documents") or []):
            for item in entry or []:
                if str(item.get("kind", "image")) == "image":
                    rows.append((index, position, item))
                    break
    return rows


def _text_rows(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    """The text-only rows named by ``--text-rows`` (documents) and ``--query-rows`` (queries).

    The image sweep cannot see them (``_pairs_rows`` selects image-carrying rows), yet the wave's own
    record puts the majority of its sub-0.999 tokens on text documents: those rows have byte-identical
    inputs on both sides (stage 1 passes), so their divergence is the text model's own -- no tower, no
    pixel affine, no render question -- and the text path has to be measured on its own.
    """
    rows: list[tuple[int, dict[str, Any]]] = []
    text_rows = _index_set(args.text_rows)
    query_rows = _index_set(args.query_rows)
    for index, row in enumerate(_read_pairs(args.pairs)):
        if index in text_rows:
            documents = row.get("documents") or [""]
            rows.append((index, {"kind": "text", "side": "document", "text": str(documents[0] or "")}))
        elif index in query_rows:
            rows.append((index, {"kind": "text", "side": "query", "text": str(row.get("query") or "")}))
    return rows


def _text_render(args: argparse.Namespace, item: dict[str, Any]) -> str:
    """The checkpoint's own text render: ``format_query`` / ``format_document`` (strip rules included)."""
    if item["side"] == "query":
        return args.query_template + item["text"].strip()
    return (args.document_prompt + item["text"]).strip()


def _read_pairs(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_image(entry: dict[str, Any]) -> Any:
    from PIL import Image

    uri = str(entry["uri"])
    if not uri.startswith("data:"):
        raise ValueError(f"the probe takes inline images; got {uri[:48]!r}")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as handle:
        return handle.convert("RGB")


def _first_divergent(report: dict[str, Any], feed_key: str) -> Any:
    """The first layer whose max-abs-diff crosses the threshold on one feed, with the in-run floor."""
    order: list[str] = []
    for row in report["rows"]:
        table = row.get(feed_key)
        if not table:
            continue
        for key in table:
            if key not in order:
                order.append(key)
    for key in order:
        worst = 0.0
        floor = 0.0
        for row in report["rows"]:
            entry = (row.get(feed_key) or {}).get(key) or {}
            worst = max(worst, float(entry.get("max_abs_diff") or 0.0))
            floor = max(floor, float(row.get("bf16_floor") or 0.0))
        if worst > max(FIRST_DIVERGENCE, 10 * floor):
            return {"layer": key, "max_abs_diff": worst, "bf16_floor": floor, "feed": feed_key}
    return {
        "layer": None,
        "feed": feed_key,
        "note": "no layer crossed the threshold on this feed",
        "bf16_floor": max((float(row.get("bf16_floor") or 0.0) for row in report["rows"]), default=0.0),
    }


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "probe-report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    control = report.get("control") or {}
    shared = report.get("first_divergent_layer") or {}
    conflated = report.get("first_divergent_layer_engine_pixels") or {}
    control_note = (
        json.dumps({k: v for k, v in control.items() if k not in ("status", "reason", "how")})
        if control.get("status") == "run"
        else str(control.get("reason", ""))
    )
    missing = sorted({key for row in report["rows"] for key in (row.get("E3_missing_tensors") or [])})
    lines = [
        "# topk-embed-v1 GPU probe",
        "",
        f"- model: `{report['model']}` @ `{report['revision']}`",
        f"- engine image: `{report['engine_image']}`",
        f"- dtype: engine `{report.get('dtype')}`, reference `{report.get('reference_dtype')}`",
        f"- control (probe vectors vs the wave's recorded vectors): `{control.get('status')}` {control_note}",
        "- engine-tower tensors recorded: "
        + (
            f"**missing {missing}** -- a first divergent layer read off this table is not trustworthy"
            if missing
            else "every expected tensor"
        ),
        f"- first divergent layer (shared-tensor feed): `{shared.get('layer')}` "
        f"(max-abs-diff {shared.get('max_abs_diff')})",
        f"- first divergent layer (engine-pixels feed, conflated): `{conflated.get('layer')}` "
        f"(max-abs-diff {conflated.get('max_abs_diff')})",
        "",
        "| pairs row | doc | grid (engine) | E1 engine-vs-checkpoint | E1 best variant vs engine | "
        "first layer (shared) | E6 min cosine | wave min | E7 first text layer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        table = row.get("E2_tower_on_reference_pixels") or {}
        floor = float(row.get("bf16_floor") or 0.0)
        first = next(
            (
                key
                for key in table
                if float((table[key] or {}).get("max_abs_diff") or 0.0) > max(FIRST_DIVERGENCE, 10 * floor)
            ),
            "-",
        )
        variants = row.get("E1_variants") or {}
        best = None
        for name, entry in variants.items():
            value = (entry.get("vs_engine") or {}).get("max_abs_diff")
            if value is not None and (best is None or value < best[1]):
                best = (name, value)
        end = row.get("E6_end_to_end") or {}
        cosine = end.get("cosine_per_token") or {}
        text_layers = row.get("E7_text_layers") or {}
        first_text = next(
            (
                key
                for key in text_layers
                if float((text_layers[key] or {}).get("max_abs_diff") or 0.0) > max(FIRST_DIVERGENCE, 10 * floor)
            ),
            "-",
        )
        lines.append(
            f"| {row['pairs_index']} | {row.get('document_index')} | {row.get('engine_grid_thw')} | "
            f"{(row.get('E1_pixel_values') or {}).get('max_abs_diff')} | "
            f"{best[0] if best else '-'}: {best[1] if best else '-'} | {first} | "
            f"{cosine.get('min')} | {cosine.get('wave_min_cosine')} | {first_text} |"
        )
    lines += ["", "## raw", "", "```json", json.dumps(report, indent=1, default=str), "```", ""]
    (out_dir / "probe-report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """The probe's own parser: the wrapper builds its argv by hand, so the flags must match here."""
    parser = argparse.ArgumentParser(description="topk-embed-v1's controlled GPU probe")
    parser.add_argument("--side", choices=["engine", "reference"], default="engine")
    parser.add_argument(
        "--check-engine",
        action="store_true",
        help="check the engine image's vLLM API (no model, no GPU) and exit",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--rows", default="")
    parser.add_argument("--text-rows", default="", help="pairs rows whose TEXT document is probed")
    parser.add_argument("--query-rows", default="", help="pairs rows whose query is probed")
    parser.add_argument("--document-prompt", default="Document: ", help="the checkpoint's document_prompt")
    parser.add_argument("--query-template", default="Query: ", help="the checkpoint's query_template")
    parser.add_argument("--image", default="")
    parser.add_argument("--text", default="", help="the reference side's text row")
    parser.add_argument("--text-side", choices=["document", "query"], default="document")
    parser.add_argument("--engine-pixels", default="")
    parser.add_argument("--recipe-dir", default=".")
    parser.add_argument("--reference-python", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="the engine's LLM dtype (the serve's dtype); the reference side has its own",
    )
    parser.add_argument(
        "--reference-dtype",
        default="bfloat16",
        help="the checkpoint's own load dtype on the reference side",
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--control-vectors", default="")
    parser.add_argument("--wave-cosines", default="")
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1310720)
    parser.add_argument("--image-token-id", type=int, default=248056)
    parser.add_argument("--max-model-len", type=int, default=8448)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_engine:
        return check_engine()
    if args.wave_cosines:
        path = Path(args.wave_cosines)
        args.wave_cosines = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if args.side == "reference":
        required = ("out", "model", "revision", "text") if args.text else ("out", "model", "revision", "image")
        for name in required:
            if not getattr(args, name):
                parser.error(f"--{name} is required on the reference side")
        return reference_side(args)
    for name in ("out", "model", "revision", "pairs", "reference_python"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required on the engine side")
    return engine_side(args)


if __name__ == "__main__":
    sys.exit(main())
