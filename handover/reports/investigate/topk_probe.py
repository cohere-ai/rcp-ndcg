#!/usr/bin/env python3
"""topk-embed-v1: the controlled GPU probe for the image-document divergence (investigate lane, Part B).

Runs INSIDE the engine image on the node (the image's python, with the recipe's plugin wheel installed
by the node bootstrap).  The engine side loads vLLM in process and the engine's own HF processor; the
reference side is this same file re-executed under the family's reference venv (sentence-transformers
over the checkpoint's trusted code), which dumps its tensors to a ``.pt`` file.  Every experiment writes
its tensors' shapes, hashes and diffs to ``probe-report.json`` and ``probe-report.md``; nothing is
compared without the numbers being in the report.

    python3 topk_probe.py --out <dir> --model <checkpoint> --revision <rev> --pairs <pairs.jsonl> \
        --reference-python <state>/reference/topk-embed-v1/bin/python [--rows 20,26,31,36]

    # the reference side (called by the engine side, never by hand):
    <reference-python> topk_probe.py --side reference --image <file.png> --model <checkpoint> \
        --revision <rev> --recipe-dir <recipe dir> --out <dump.pt>

The experiments (the report's Part B.1 table): E1 preprocessing isolation; E2 the same tensors into
both vision towers; E3 layer-by-layer dumps; E4 the merged image-token embeddings; E5 cross-feed; E6
the grid sweep; E7 the attention window.  A failed experiment is recorded with its traceback and the
run continues: the probe is a measurement, not a gate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

DEFAULT_ROWS: list[int] = []
"""Empty: every pairs row carrying an image document (the nine media rows of the shipped pairs file)."""

FIRST_DIVERGENCE = 1e-2
"""A tensor's max-abs-diff above this names the layer (the vision tower's bf16 floor is ~1e-3)."""


def sha256(tensor: Any) -> str:
    """The tensor's bytes' hash (dtype and shape included), so equal tensors hash equal."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(tensor))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()[:32]


def stats(left: Any, right: Any) -> dict[str, Any]:
    """The comparison of two tensors: shapes, dtypes, hashes, max/mean abs diff and cosine."""
    import numpy as np

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    out: dict[str, Any] = {
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "left_sha256": sha256(left),
        "right_sha256": sha256(right),
        "equal": bool(a.shape == b.shape and np.array_equal(a, b)),
    }
    if a.shape != b.shape:
        out["note"] = "shapes differ: no elementwise comparison"
        return out
    diff = np.abs(a - b)
    out["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
    out["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
    na = float(np.linalg.norm(a.ravel()))
    nb = float(np.linalg.norm(b.ravel()))
    if na and nb:
        out["cosine"] = float(np.dot(a.ravel(), b.ravel()) / (na * nb))
    return out


def _tensor(tensor: Any) -> Any:
    """One tensor on the host, detached, float32 (numpy conversion must never see CUDA)."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor


def _hooks(module: Any, names: dict[str, Any], sink: dict[str, Any]) -> list[Any]:
    """Forward hooks over ``names`` (attribute path -> report key), recording each output in ``sink``."""
    handles = []

    def make(key: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, tuple):
                output = output[0]
            sink[key] = _tensor(output)

        return record

    for path, key in names.items():
        target = module
        for attribute in path.split("."):
            target = getattr(target, attribute)
        handles.append(target.register_forward_hook(make(key)))
    return handles


# ---------------------------------------------------------------------------
# the reference side (this file under the reference venv)
# ---------------------------------------------------------------------------


def reference_side(args: argparse.Namespace) -> int:
    """Load the checkpoint's own path, dump its tensors for one image, write ``args.out``."""
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
        model_kwargs={"dtype": "bfloat16"},
    )
    module = next(m for m in model.modules() if type(m).__name__ == "TopkEmbedTransformer")
    topk = next(m for m in module.modules() if type(m).__name__ == "TopkEmbedModel")
    visual = topk.model.visual
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
    layers: dict[str, Any] = {}
    handles = _hooks(
        visual,
        {"patch_embed": "patch_embed", "merger": "merger"}
        | {f"blocks.{index}": f"block_{index}" for index in range(len(visual.blocks))},
        layers,
    )
    try:
        with torch.no_grad():
            merged = visual(row.pixel_values, **vision_inputs)
    finally:
        for handle in handles:
            handle.remove()
    layers["merged"] = _tensor(merged)

    with torch.no_grad():
        document = model.encode_document([image])[0]
    torch.save(
        {
            "grid_thw": row.image_grid_thw,
            "target": (target_h, target_w),
            "layers": layers,
            "variants": {key: _tensor(value) for key, value in variants.items()},
            "document_vectors": _tensor(document),
        },
        args.out,
    )
    print(f"reference: {args.image} -> {args.out} ({len(layers)} tensors)")
    return 0


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


def engine_model(llm: Any) -> Any:
    """The loaded vLLM model object, through the paths the in-process engine has used."""
    candidates = [
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
    ]
    for path in candidates:
        target = llm
        for attribute in path:
            target = getattr(target, attribute, None)
            if target is None:
                break
        if target is not None:
            return target
    for module in llm.llm_engine.named_modules():
        if type(module).__name__ in ("TopkEmbedModel", "ColQwen3_5Model"):
            return module
    raise RuntimeError("cannot reach the engine's model object: the in-process accessor path changed")


def engine_visual(model: Any) -> Any:
    """The engine's vision tower (``visual`` on the model or on its inner model)."""
    visual = getattr(model, "visual", None)
    if visual is None:
        visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise RuntimeError("cannot reach the engine's vision tower")
    return visual


def engine_side(args: argparse.Namespace) -> int:
    """Load vLLM in process, run E1-E7 for the requested rows, write the report."""
    import torch
    from transformers import AutoProcessor
    from vllm import LLM

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "engine_image": os.environ.get("RCP_IMAGE", "unknown"),
        "device": args.device,
        "experiments": {},
        "rows": [],
    }
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    image_processor = processor.image_processor
    report["experiments"]["E1_preprocessing"] = {
        "hypothesis": "the engine's PIL BICUBIC + fp32 affine pixels differ from the checkpoint's "
        "torchvision antialias BICUBIC + bf16 affine pixels",
        "engine_processor": type(image_processor).__name__,
        "pin": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
    }

    llm = LLM(
        model=args.model,
        revision=args.revision,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"images_kwargs": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}},
        hf_overrides={"document_keep_token_ids": [args.image_token_id]},
    )
    model = engine_model(llm)
    visual = engine_visual(model)
    report["experiments"]["E1_preprocessing"]["engine_visual"] = type(visual).__name__

    rows = _pairs_rows(args)
    reference_python = args.reference_python
    for row_index, image in rows:
        row_report: dict[str, Any] = {"pairs_index": row_index, "image": image["sha256"][:16]}
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
            # the reference side, in its own venv.
            image_path = out_dir / f"row{row_index}.png"
            pil.save(image_path)
            dump_path = out_dir / f"row{row_index}-reference.pt"
            subprocess.run(
                [
                    reference_python,
                    str(Path(__file__).resolve()),
                    "--side",
                    "reference",
                    "--image",
                    str(image_path),
                    "--model",
                    args.model,
                    "--revision",
                    args.revision,
                    "--recipe-dir",
                    args.recipe_dir,
                    "--device",
                    args.device,
                    "--out",
                    str(dump_path),
                ],
                check=True,
            )
            reference = torch.load(dump_path, map_location="cpu", weights_only=False)
            row_report["reference_grid_thw"] = reference["grid_thw"].tolist()
            row_report["reference_target_hw"] = list(reference["target"])
            # E1: the pixel tensors.
            row_report["E1_pixel_values"] = stats(engine_pixels, reference["variants"]["pixel_values_checkpoint"])
            row_report["E1_variants"] = {
                name: {
                    "shape": list(value.shape),
                    "sha256": sha256(value),
                    "vs_checkpoint": stats(value, reference["variants"]["pixel_values_checkpoint"]),
                }
                for name, value in reference["variants"].items()
            }
            # E2/E3/E4: the towers, on the engine's own pixels and on the reference's.
            for feed, pixels in (
                ("engine_pixels", engine_pixels),
                ("reference_pixels", reference["variants"]["pixel_values_checkpoint"]),
            ):
                dumps: dict[str, Any] = {}
                handles = _hooks(
                    visual,
                    {"patch_embed": "patch_embed", "merger": "merger"}
                    | {f"blocks.{index}": f"block_{index}" for index in range(len(visual.blocks))},
                    dumps,
                )
                try:
                    with torch.no_grad():
                        merged = visual(
                            pixels.to(device=args.device, dtype=visual.patch_embed.proj.weight.dtype),
                            engine_grid.to(args.device),
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                dumps["merged"] = _tensor(merged)
                row_report[f"E2_tower_on_{feed}"] = {
                    key: stats(value, reference["layers"].get(key)) for key, value in dumps.items()
                }
            # E4: the image-token embeddings the text model receives (the merger output).
            row_report["E4_image_token_embeddings"] = stats(_tensor(merged), reference["layers"]["merged"])
            # the engine's end-to-end token vectors for the same image, and the reference's.
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            engine_vectors = _engine_encode(llm, rendered, pil)
            reference_vectors = reference["document_vectors"]
            row_report["E6_end_to_end"] = {
                "engine_tokens": int(len(engine_vectors)),
                "reference_tokens": int(len(reference_vectors)),
                "cosine_per_token": _per_token_cosine(engine_vectors, reference_vectors),
            }
            # E7: the attention window, on this row only (mutate the text config, re-encode).
            row_report["E7_window"] = _window_experiment(llm, rendered, pil, reference_vectors)
            row_report["passed"] = True
        except Exception as error:  # noqa: BLE001 - one failed row must not stop the sweep
            row_report["passed"] = False
            row_report["error"] = f"{type(error).__name__}: {error}"
            row_report["traceback"] = traceback.format_exc()
        report["rows"].append(row_report)
        _write_report(out_dir, report)

    report["first_divergent_layer"] = _first_divergent(report)
    _write_report(out_dir, report)
    return 0


def _engine_encode(llm: Any, prompt: str, image: Any) -> Any:
    """One image document through the engine's own encode (the wave's signal, in process)."""
    import numpy as np

    outputs = llm.encode(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        pooling_task="token_embed",
        use_tqdm=False,
    )
    return np.asarray(outputs[0].outputs.data, dtype=np.float64)


def _per_token_cosine(engine_vectors: Any, reference_vectors: Any) -> dict[str, Any]:
    """The per-token cosine between the engine's kept vectors and the reference's kept vectors."""
    import numpy as np

    engine = np.asarray(engine_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if engine.shape != reference.shape:
        return {
            "note": "token counts differ",
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


def _window_experiment(llm: Any, prompt: str, image: Any, reference_vectors: Any) -> dict[str, Any]:
    """E7: one row with the checkpoint's sliding window disabled in the engine's text config.

    The window is the text config's; disabling it means replacing every layer type with full attention
    and clearing ``sliding_window`` on the loaded config, then re-encoding the same image.  The
    unpatched re-encode is measured too (the determinism floor).  The signal is whether the
    divergence from the reference MOVES: if it does not, the window semantics are excluded.
    """
    before = _per_token_cosine(_engine_encode(llm, prompt, image), reference_vectors)
    try:
        config = llm.llm_engine.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        layer_types = list(getattr(text_config, "layer_types", []) or [])
        saved = {
            "sliding_window": getattr(text_config, "sliding_window", None),
            "layer_types": layer_types,
        }
        if layer_types:
            text_config.layer_types = [
                "full_attention" if str(kind) != "full_attention" else kind for kind in layer_types
            ]
        if hasattr(text_config, "sliding_window"):
            text_config.sliding_window = None
        after = _per_token_cosine(_engine_encode(llm, prompt, image), reference_vectors)
        text_config.sliding_window = saved["sliding_window"]
        if layer_types:
            text_config.layer_types = saved["layer_types"]
        return {"before": before, "with_window_disabled": after, "restored": saved}
    except Exception as error:  # noqa: BLE001 - a config shape change must be recorded, not fatal
        return {"before": before, "error": f"{type(error).__name__}: {error}"}


def _pairs_rows(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    """The pairs rows carrying an image document (or the ``--rows`` selection), with their entries."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(_read_pairs(args.pairs)):
        if args.rows and index not in args.rows:
            continue
        for entry in (row.get("media") or {}).get("documents") or []:
            for item in entry or []:
                if str(item.get("kind", "image")) == "image":
                    rows.append((index, item))
                    break
    return rows


def _read_pairs(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_image(entry: dict[str, Any]) -> Any:
    from PIL import Image

    uri = str(entry["uri"])
    if not uri.startswith("data:"):
        raise ValueError(f"the probe takes inline images; got {uri[:48]!r}")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as handle:
        return handle.convert("RGB")


def _first_divergent(report: dict[str, Any]) -> Any:
    """The first layer whose max-abs-diff crosses ``FIRST_DIVERGENCE`` on the engine's own pixels."""
    order: list[str] = []
    for row in report["rows"]:
        table = row.get("E2_tower_on_engine_pixels")
        if not table:
            continue
        for key in table:
            if key not in order:
                order.append(key)
    for key in order:
        worst = 0.0
        for row in report["rows"]:
            entry = (row.get("E2_tower_on_engine_pixels") or {}).get(key) or {}
            worst = max(worst, float(entry.get("max_abs_diff") or 0.0))
        if worst > FIRST_DIVERGENCE:
            return {"layer": key, "max_abs_diff": worst}
    return {"layer": None, "note": "no layer crossed the threshold on the engine's own pixels"}


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "probe-report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    lines = [
        "# topk-embed-v1 GPU probe",
        "",
        f"- model: `{report['model']}` @ `{report['revision']}`",
        f"- engine image: `{report['engine_image']}`",
        f"- first divergent layer (engine pixels): `{(report.get('first_divergent_layer') or {}).get('layer')}`",
        "",
        "| pairs row | grid (engine) | E1 pixel max-abs-diff | first layer diff | E6 min cosine | E7 window |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        table = row.get("E2_tower_on_engine_pixels") or {}
        first = next(
            (key for key in table if float((table[key] or {}).get("max_abs_diff") or 0.0) > FIRST_DIVERGENCE),
            "-",
        )
        end = row.get("E6_end_to_end") or {}
        cosine = (end.get("cosine_per_token") or {}).get("min")
        window = row.get("E7_window") or {}
        moved = None
        if "before" in window and "with_window_disabled" in window:
            moved = (window["before"].get("min"), window["with_window_disabled"].get("min"))
        lines.append(
            f"| {row['pairs_index']} | {row.get('engine_grid_thw')} | "
            f"{(row.get('E1_pixel_values') or {}).get('max_abs_diff')} | {first} | {cosine} | {moved} |"
        )
    lines += ["", "## raw", "", "```json", json.dumps(report, indent=1, default=str), "```", ""]
    (out_dir / "probe-report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="topk-embed-v1's controlled GPU probe")
    parser.add_argument("--side", choices=["engine", "reference"], default="engine")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--rows", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--recipe-dir", default=".")
    parser.add_argument("--reference-python", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=1310720)
    parser.add_argument("--image-token-id", type=int, default=248056)
    parser.add_argument("--max-model-len", type=int, default=8448)
    args = parser.parse_args(argv)
    if args.side == "reference":
        return reference_side(args)
    if not args.pairs or not args.reference_python:
        parser.error("--pairs and --reference-python are required on the engine side")
    args.rows = {int(value) for value in args.rows.split(",") if value.strip()} if args.rows else set()
    return engine_side(args)


if __name__ == "__main__":
    sys.exit(main())
