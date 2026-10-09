"""Test oracle: the reference processors' image resize and frame sampling, vendored verbatim.

Copied from the Apache-2.0 sources below at the named commits (attributed in NOTICE) so the tests check the client's
preparation against what the engine runs, without a dependency on transformers or vLLM. Only formatting is
changed (and the parts that read an engine's own objects are reduced to the numbers they read).

* transformers 528c26713c8f3774fb56d409da766bc95452aafb,
  src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py:63-89 (``smart_resize``): the resize of every Qwen-VL
  image processor, which vLLM calls with ``factor = patch_size * merge_size`` and the processor's ``size``
  (vllm/model_executor/models/qwen3_vl.py:957-1003, qwen2_vl.py:952-978).
* vLLM 3627a6a124896edbc9f802e7c7e120633007e29d, vllm/multimodal/video.py:214-240
  (``VideoBackend.compute_frames_index_to_sample``, the default ``opencv`` loader).
"""

from __future__ import annotations

import math

import numpy as np


# --- transformers: models/qwen2_vl/image_processing_qwen2_vl.py:63-89 ---------------------------------------------
def hf_smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# --- vLLM: vllm/multimodal/video.py:214-240 (VideoBackend.compute_frames_index_to_sample) ---------------------------
def vllm_frame_indices(total_frames_num: int, duration: float, num_frames: int, fps: float) -> list[int]:
    num_frames_to_sample = total_frames_num
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, total_frames_num)
    if fps > 0:
        num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
    num_frames_to_sample = max(1, num_frames_to_sample)

    if num_frames_to_sample == total_frames_num:
        return list(range(num_frames_to_sample))
    return np.linspace(0, total_frames_num - 1, num_frames_to_sample, dtype=int).tolist()


# --- What vLLM applies to an image when started without media flags -----------------------------------------------
# (factor, min_pixels, max_pixels) per family. vLLM takes the checkpoint's preprocessor_config.json
# (qwen3_vl.py:982-985 `default_size=image_processor.size`). Checkpoint values from the public
# preprocessor_config.json of Qwen/Qwen2-VL-7B-Instruct and Qwen/Qwen2.5-VL-7B-Instruct
# ({min,max}_pixels 3136, 12845056; patch 14, merge 2) and of Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3.5-397B-A17B(-FP8)
# and Qwen/Qwen3.6-27B-FP8 (size 65536..16777216; patch 16, merge 2).
ENGINE_DEFAULTS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("vllm", "qwen2_vl"): (28, 3136, 12845056),
    ("vllm", "qwen2_5_vl"): (28, 3136, 12845056),
    ("vllm", "qwen3_vl"): (32, 65536, 16777216),
}
