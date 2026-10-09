# The judge catalog as a spec (workstream 08 C, lane `l08-cat`)

Status: research only, 2026-10-09. This file is the deliverable of lane `l08-cat`; the recipe files themselves are
lane `l08-judges` (08 B+D). It fixes, for each of the six judges of decision 15, every fact that lane needs: the Hub
repo and 40-hex revision, licence and gated status, the checkpoint's `config.json` architectures, the vLLM v0.31.0
registry line that serves each architecture, the serve block translated from the paper's engine settings into vLLM
flags (each flag cited in the vLLM v0.31.0 source), the client block from today's presets, `resources.gpus` with the
memory arithmetic, and the family grouping under decision 34. Open questions are in the last section.

Sources and method:

- Hub facts resolved anonymously through the public Hub API (`https://huggingface.co/api/models/<repo>`, the repo's
  `config.json`, `hf_quant_config.json`, `preprocessor_config.json`, `tokenizer_config.json`,
  `model.safetensors.index.json` and the model card at the pinned revision), checked **2026-10-09**. The three
  Qwen3.8 pins are the ones `rcp-ndcg-test/scenarios/{identity,outage,text-four-phases,vidore}.yaml` already carry
  (checked 2026-10-07 there); all six pins were re-verified against the Hub API on 2026-10-09.
- vLLM cloned at tag `v0.31.0` (commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`) into the lane scratch; every
  `vllm/...` citation below is `file:line` at that tag. Registry citations are
  `vllm/model_executor/models/registry.py`.
- Paper-side engine settings: `experiments/paper/serve/*.sglang.sh` (the paper's SGLang commands) and
  `experiments/paper/README.md`; client blocks: `rcp-ndcg/src/rcp_ndcg/judging/judges/*.yaml` (the self-hosted
  presets) and the four T4 scenarios.
- KV arithmetic below counts only the full-attention layers: all five Qwen checkpoints are hybrids
  (`full_attention_interval: 4` — every fourth layer is full attention, the rest linear attention), and their
  linear-attention layers carry a fixed per-sequence recurrent state, not per-token KV. `gpt-oss-120b` is dense +
  sliding-window attention (window 128 on every second layer, `layer_types` in its `config.json`).

---

## 1. The catalog at a glance and the family grouping (decision 34)

Decision 34 (`drafts/recipe-families.md`): one family directory per model family with a `family.yaml` holding the
shared client/serve blocks and a `variants` table holding only per-variant facts; every variant resolves to its own
full recipe id. For the judges this yields five families, six recipes:

| Family | Variants (recipe ids) | Why grouped / not grouped |
|---|---|---|
| `qwen3.5-397b` | `qwen3.5-397b-a17b-nvfp4` | Single variant: the FP8 sibling is dropped (`qwen35_397b_fp8`, decision 15 — the paper never ran it). |
| `gpt-oss-120b` | `gpt-oss-120b` | Singleton. Distinct architecture (`GptOssForCausalLM`), dense+SWA attention, MXFP4, vocab 201 088 — shares nothing model-specific with the Qwen judges beyond the catalog-wide block. |
| `qwen3.6-27b` | `qwen3.6-27b-fp8` | Singleton (the paper's TREC-DL judge). |
| `qwen3.8-27b` | `qwen3.8-27b-fp8` | Singleton (the T4 identity/outage judge). |
| `qwen3.8-flash-next` | `qwen3.8-flash-next-nvfp4`, `qwen3.8-flash-next-fp8` | The one multi-variant family: the same model in two quantisations. |

Evidence for the groupings, checked at the pinned revisions:

- **The two Flash-Next variants are one model.** Identical serving-relevant text configs (`config.json`
  `text_config`: 48 hidden layers, hidden 2560, 512 experts, top-10, moe intermediate 640, vocab 248 320, max
  positions 262 144, head_dim 256, 2 KV heads, `full_attention_interval` 4; the raw JSON differs in three inert
  keys only — `norm_topk_prob`, which defaults to the same value in vLLM's `Qwen4ExpTextConfig`,
  `vllm/transformers_utils/configs/qwen3_next.py:214`, and `number_of_conv_states`/`seed`, which the engine does
  not read); byte-identical `tokenizer_config.json` and chat template
  (sha256 prefix `b11349aafa7cdc6a` / `c3cf9e34abf4f9e3`); identical `preprocessor_config.json` (patch 16, merge 2,
  temporal 2, size shortest 65 536 / longest 16 777 216 px). They differ only where the spec declares per-variant
  facts: repo, revision, quantisation (`modelopt_fp4` vs the checkpoint's own `fp8`), licence and the GPU class the
  quantisation implies.
- **The two 27B judges stay separate.** `Qwen/Qwen3.6-27B-FP8` and `Qwen/Qwen3.8-27B-FP8` are architecturally
  identical (both `Qwen3_5ForConditionalGeneration`; their `config.json` files differ only in
  `transformers_version`) — but their `tokenizer_config.json` files are not byte-identical and their chat templates
  differ materially: Qwen3.8's template adds reasoning-effort instructions (xhigh default, medium/low) and changes
  how assistant turns preserve prior thinking. They are also different Hub generations (Qwen3.6 vs Qwen3.8) with
  different provenance (the paper's TREC-DL judge vs the owner's T4 judge). Under decision 34's rule — anything that
  differs beyond the declared per-variant fields is not forced into a family — they stay two families. (If the owner
  prefers one `qwen3.8` super-family over three, the blocks below separate cleanly either way; see open question 7.)

**Shared across the whole catalog** (every family's `family.yaml`, not restated per variant below):

- `role: judge`, `client.api: chat`, no `reference` (decision 15); the client block is rcp-ndcg's judge config,
  validated by rcp-ndcg (R30).
- `decoding: json_schema`: the server constrains each answer to the stage's JSON schema (response_format
  json_schema, strict). Serve the model with its reasoning parser so the schema constrains the answer, not the
  reasoning; vLLM applies the grammar only after reasoning ends (`vllm/v1/structured_output/__init__.py:240-262`).
- No temperature is sent: the server's default sampling, as the paper's runs used (`temperature: null` in every
  preset and scenario).
- Engine: the stock image `vllm/vllm-openai:v0.31.0` only (decision 2); SGLang is retired (workstream 08 A). Every
  judge's architecture is in vLLM v0.31.0's registry (section 4).
- The recipe pins `revision` and the client's `tokenizer` at the same 40-hex commit (the `repo@revision` shape the
  qwen3.5 preset and the scenarios already use; the gpt-oss and qwen3.6 presets name the tokenizer bare — the recipe
  pins it).

---

## 2. Flag translation: paper settings to vLLM v0.31.0

One home for the vLLM citations; the per-family serve blocks in section 3 reference these lines. Argument flag
definitions are in `vllm/engine/arg_utils.py` (field + `add_argument` lines):

| vLLM flag (value) | Paper-side source | vLLM v0.31.0 citation |
|---|---|---|
| `--revision <sha>` | `--revision` in both sglang.sh scripts | `arg_utils.py:946` (field `:606`; `vllm/config/model.py:210`) |
| `--served-model-name <name>` | `--served-model-name` (sglang.sh:14); the scenarios serve every judge as `judge` | `arg_utils.py:988` (field `:473`) |
| `--tensor-parallel-size N` | `--tensor-parallel-size 4` (sglang.sh:18); `1`/`4` in the scenarios | `arg_utils.py:1151` (field `:523`) |
| `--data-parallel-size N` | `--data-parallel-size 2` (sglang.sh:18, the paper's second replica group) | `arg_utils.py:1183` (field `:530`; DP world-size math `:2296-2351`) |
| `--quantization modelopt_fp4` | `--quantization modelopt_fp4` (qwen35 sglang.sh:20; scenarios text-four-phases.yaml:46-47, vidore.yaml:45-46) | `arg_utils.py:952` (field `:614`); `ModelOptNvFp4Config.get_name() -> "modelopt_fp4"` (`layers/quantization/modelopt.py:715,746`); an `hf_quant_config.json` with `quant_algo: NVFP4` maps to `modelopt_fp4` (`modelopt.py:138`); docs `features/quantization/modelopt.md:19` |
| `--kv-cache-dtype fp8_e4m3` | `--kv-cache-dtype fp8_e4m3` (qwen35 sglang.sh:20) | `arg_utils.py:1334` (field `:493`); the literal is in `CacheDType` (`config/cache.py:39-46`); default `auto` (`config/cache.py:123`) |
| `--gpu-memory-utilization 0.85` | SGLang `--mem-fraction-static 0.85` (qwen35 sglang.sh:21); **renamed in vLLM** — no `--mem-fraction-static` exists at v0.31.0 | `arg_utils.py:1326-1329` (field `:588`); default 0.92 (`config/cache.py:103`) |
| `--max-model-len N` | SGLang `--context-length` (gpt-oss sglang.sh:20 = 131 072; qwen35 sglang.sh:23 = 262 144); the scenarios pass `--max-model-len` directly | `arg_utils.py:951` (field `:495`) |
| `--reasoning-parser <name>` | SGLang `--reasoning-parser qwen3` / `gpt-oss`; the scenarios pass `qwen3` | `arg_utils.py:1093` (field `:702`); parser registry: `"qwen3"` → `qwen3_engine_reasoning_parser` / `Qwen3ParserReasoningAdapter` (`reasoning/__init__.py:143-145`, lazy registration `:15`), `"openai_gptoss"` → `gptoss_reasoning_parser` / `GptOssReasoningParser` (`reasoning/__init__.py:71-73`) — **vLLM's name for the gpt-oss parser is `openai_gptoss`, not SGLang's `gpt-oss`**; vLLM also auto-selects it for gpt-oss checkpoints when unset (`models/config.py:423-425`), but the recipe declares it explicitly |
| `--limit-mm-per-prompt '{"image": 10}'` | Not in the paper's SGLang commands (SGLang derived media limits from the processor); the T4 scenarios declare it | `arg_utils.py:1432` (field `:625`); without it each modality defaults to 999 (`config/multimodal.py:127-144`) — the recipe declares the cap so it matches the client's `max_images` |

No `--quantization` flag is passed where the checkpoint quantises itself and vLLM detects it from
`config.json`: `quant_method: "fp8"` → `Fp8Config` (`layers/quantization/fp8.py:92,138-139`) for
`qwen3.6-27b-fp8`, `qwen3.8-27b-fp8` and `qwen3.8-flash-next-fp8`; `quant_method: "mxfp4"` is rewritten to
`gpt_oss_mxfp4` and served natively for `gpt-oss-120b` (`models/config.py:414-421`).

**SGLang knobs dropped in the translation** (declared, per the nothing-silently-defaulted rule):

- Scheduling/concurrency (`--max-running-requests`, `--chunked-prefill-size`, `--max-prefill-tokens`,
  `--stream-interval`, `--tokenizer-worker-num`, `--num-continuous-decode-steps`, `--cuda-graph-max-bs`,
  `--schedule-policy lpm`): scheduler capacity tuning; it cannot change a judgement, and the client's `concurrency`
  is the judge-side bound. vLLM's equivalents (`--max-num-seqs`, `--max-num-batched-tokens`) stay at their defaults.
- Blackwell kernel selection (`--attention-backend trtllm_mha`, `--moe-runner-backend flashinfer_trtllm`,
  `--fp4-gemm-backend flashinfer_cutlass`, `--enable-flashinfer-allreduce-fusion`): vLLM selects the NVFP4 GEMM
  kernel per platform at load time (`docs/features/quantization/modelopt.md:24-40`) and picks attention backends
  itself.
- `--page-size 64` (qwen35 sglang.sh:32): SGLang's hybrid-cache page size; vLLM sizes its own blocks
  (`--block-size`, `config/cache.py`), and the hybrid models' Mamba block grid is engine-managed.
- `--disable-radix-cache` + `--mamba-scheduler-strategy no_buffer`: SGLang hybrid-cache workarounds. vLLM v0.31.0
  serves the hybrid checkpoints with prefix caching on by default (`config/cache.py:142`,
  `enable_prefix_caching: bool = True`) and its Mamba cache mode defaults to `align` when prefix caching is enabled
  (`config/cache.py:190-198`).
- `--tool-call-parser qwen3_coder` (qwen35 sglang.sh:34): the judge never requests tool calls; the parser is for
  tool-call replies, not the judge path.
- `--model-loader-extra-config`, `--enable-metrics*`: loader/metrics tuning, not judgement-relevant.

---

## 3. The five families in detail

Every variant's `variants` row carries: `id`, `model` (Hub repo), `revision`, and the per-variant overrides listed
below. Everything else is the family's shared block. `resources.gpus` is the per-replica count, followed by the
paper's whole-node shape where the paper ran more than one replica.

### 3.1 Family `qwen3.5-397b` — the paper's primary judge

**Shared (family.yaml):** served name `qwen3.5-397b`; client block from the preset
`judging/judges/qwen35_397b_nvfp4.yaml`: `temperature: null`, `max_output_tokens: 16384`, `context_tokens: 262144`,
`tokenizer: nvidia/Qwen3.5-397B-A17B-NVFP4@0368c1b3…`, `concurrency: 256`, `decoding: json_schema`,
`image_processor: qwen3_vl`.

**Variant `qwen3.5-397b-a17b-nvfp4`:**

| Fact | Value | Source |
|---|---|---|
| Hub repo | `nvidia/Qwen3.5-397B-A17B-NVFP4` | preset + `experiments/paper/serve/qwen35_397b_nvfp4.sglang.sh:12` |
| Revision (pin) | `0368c1b3233414cd4a617b8ff9515e25752dc16c` — the paper's own pin; **not** the Hub head (`12b28061efd8145b11226e4fab2528ee4f39ef09`, repo last modified 2026-06-30; pin re-verified to resolve at the API on 2026-10-09) | sglang.sh:13; Hub API 2026-10-09 |
| Licence / gated | Apache-2.0 / not gated | card + Hub API `gated: false` |
| `config.json` architectures | `Qwen3_5MoeForConditionalGeneration` (MoE: 60 layers, hidden 4096, 512 experts, top-10, moe intermediate 1024, vocab 248 320, mpe 262 144, head_dim 256, 2 KV heads) | `config.json` at the pin |
| vLLM registry line | `Qwen3_5MoeForConditionalGeneration` → `registry.py:597-600` (module `qwen3_5`, `vllm/model_executor/models/qwen3_5.py`) | tag v0.31.0 |
| Weights | 251.1 GB (251 135 201 368 B, 11 safetensors shards — `model.safetensors.index.json` `metadata.total_size`); NVFP4 (ModelOpt; `hf_quant_config.json` `quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`) | Hub at the pin |

**Serve block** (translated from `qwen35_397b_nvfp4.sglang.sh:12-34`; flags per section 2):

```bash
vllm serve nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --revision 0368c1b3233414cd4a617b8ff9515e25752dc16c \
  --served-model-name qwen3.5-397b \
  --tensor-parallel-size 4 --data-parallel-size 2 \
  --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 10}'
```

**resources.gpus: 8 as the paper served it (TP4 × DP2; one replica = 4 GPUs).** Arithmetic: weights
251.1 GB / 4 = 62.8 GB per GPU; at `--gpu-memory-utilization 0.85` on a 192 GB B200 that leaves
163.2 − 62.8 ≈ 100 GB per GPU for cache. KV (FP8): 15 full-attention layers × 2 × 2 KV heads × 256 head dim
= 15 360 B/token across ranks; at TP4 the 2 KV heads replicate ×2, so 7 680 B/token per GPU (a full-length
262 144-token sequence ≈ 1.9 GiB per GPU). Linear-attention recurrent state (fp32,
`mamba_ssm_dtype: float32`): 45 linear layers × 64 value heads × 128 × 128 × 4 B ≈ 189 MB per sequence, 47 MB per
GPU at TP4 (value heads split). The 80 GB class fits only with a much smaller `--max-model-len`; native FP4 GEMM
wants SM100 anyway (Hopper falls back to Marlin W4A16, `docs/features/quantization/modelopt.md:24-40`).

### 3.2 Family `gpt-oss-120b` — the paper's second judge

**Shared = the single variant.** Client block from `judging/judges/gpt_oss_120b.yaml`: model `gpt-oss-120b`,
`temperature: null`, `max_output_tokens: 8192`, `context_tokens: 131072`,
`tokenizer: openai/gpt-oss-120b`, `concurrency: 256`, `decoding: json_schema`; **no image processor** (text-only:
the checkpoint has no vision config).

| Fact | Value | Source |
|---|---|---|
| Hub repo | `openai/gpt-oss-120b` | preset + `experiments/paper/serve/gpt_oss_120b.sglang.sh:12` |
| Revision (pin) | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` — the paper's pin and the Hub head (verified equal 2026-10-09) | sglang.sh:13; Hub API 2026-10-09 |
| Licence / gated | Apache-2.0 / not gated | card + Hub API |
| `config.json` architectures | `GptOssForCausalLM` (36 layers, hidden 2880, head_dim 64, 8 KV heads, 128 local experts, top-4, vocab 201 088, mpe 131 072 via YARN factor 32; alternating sliding-window/full attention, window 128) | `config.json` at the pin |
| vLLM registry line | `GptOssForCausalLM` → `registry.py:123` (module `gpt_oss`, `vllm/model_executor/models/gpt_oss.py`) | tag v0.31.0 |
| Weights | 65.2 GB on disk (65 248 815 744 B over the index's 15 shards — `model.safetensors.index.json` `metadata.total_size`: 57.3 GB packed MXFP4 + 3.6 GB E8M0 scales + 4.3 GB BF16, for 116 829 156 672 parameters; the `original/` BF16 copies in the repo are not loaded) | Hub API + index at the pin, 2026-10-09 |

**Serve block** (translated from `gpt_oss_120b.sglang.sh:11-25`; note the parser name change to `openai_gptoss`):

```bash
vllm serve openai/gpt-oss-120b \
  --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 4 --data-parallel-size 2 \
  --max-model-len 131072 \
  --reasoning-parser openai_gptoss
```

No `--quantization`: MXFP4 is detected from `config.json` (`models/config.py:414-421`). No `--kv-cache-dtype`: the
paper set none; `auto` gives BF16 KV. No media flags: a text-only judge (no `image_processor` in the preset).

**resources.gpus: 8 as the paper served it (TP4 × DP2; one replica = 4 GPUs).** Arithmetic: weights
65.2 GB / 4 ≈ 16.3 GB per GPU; on the 80 GB class at the default 0.92 that leaves ≈ 57 GB per GPU for cache.
KV (BF16): 18 full-attention layers × 2 × 8 KV heads × 64 × 2 B = 36 864 B/token across ranks; at TP4 the 8 KV
heads split with no replication (2 per GPU), so 9 216 B/token/GPU (a full-length 131 072-token sequence ≈ 1.125 GiB
per GPU); the 18 sliding-window layers hold at most the 128-token window each (≈ 4.7 MB per sequence, negligible).
No linear-attention state. The 80 GB class is comfortable (≈ 47 full-length sequences); Hopper serves MXFP4
natively.

### 3.3 Family `qwen3.6-27b` — the paper's TREC-DL judge

**Shared = the single variant.** Client block from `judging/judges/qwen36_27b_fp8.yaml`: model `qwen3.6-27b-fp8`,
`revision: null` (the paper's checkpoint revision is not recorded), `temperature: null`,
`max_output_tokens: 16384`, **no `context_tokens`** — with no context the judge's text policy sends documents whole
(`rcp_ndcg.judging.judging`, `window_tokens`: `None` when the judge declares no context), `tokenizer:
Qwen/Qwen3.6-27B-FP8`, `concurrency: 256`, `decoding: json_schema`, `image_processor: qwen3_vl`.

| Fact | Value | Source |
|---|---|---|
| Hub repo | `Qwen/Qwen3.6-27B-FP8` | preset; `experiments/paper/README.md:40-42` |
| Revision (pin) | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` — the Hub head, resolved 2026-10-09 (the paper's revision was never recorded; open question 1) | Hub API 2026-10-09 |
| Licence / gated | Apache-2.0 / not gated | card + Hub API |
| `config.json` architectures | `Qwen3_5ForConditionalGeneration` (text config 64 layers, hidden 5120, head_dim 256, 24 query/4 KV heads, vocab 248 320, mpe 262 144, FP8 block [128,128]; vision tower depth 27, hidden 1152, patch 16, merge 2) | `config.json` at the pin |
| vLLM registry line | `Qwen3_5ForConditionalGeneration` → `registry.py:596` (module `qwen3_5`, `vllm/model_executor/models/qwen3_5.py`) | tag v0.31.0 |
| Weights | 30.9 GB (24.7 GB FP8 + 6.2 GB BF16; Hub API `safetensors`) | Hub API 2026-10-09 |

**Serve block.** The paper recorded only the served name and the reasoning parser
(`experiments/paper/README.md:40-42`); everything else below is this spec's translation (open question 1):

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --served-model-name qwen3.6-27b-fp8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

`--max-model-len 262144` is the checkpoint's own `text_config.max_position_embeddings`. No `--quantization` (FP8
auto-detected, `fp8.py:92,138-139`); no `--kv-cache-dtype` (auto → BF16 KV). Media: the checkpoint is natively
multimodal; today's preset sets no `max_images` (run configs set it — open question 3); when a run sends images the
engine cap is `--limit-mm-per-prompt '{"image": <max_images>}'` (`arg_utils.py:1432`).

**resources.gpus: 1 (TP1; this spec's assumption — 27B-class FP8 fits an 80 GB GPU).** Arithmetic: weights
30.9 GB; at the default 0.92 on 80 GB that leaves ≈ 43 GB for cache. KV (BF16): 16 full-attention layers × 2 ×
4 KV heads × 256 × 2 B = 65 536 B/token (a 131 072-token sequence ≈ 8 GiB); linear-attention recurrent state:
48 linear layers × 48 value heads × 128 × 128 × 4 B ≈ 151 MB per sequence. `--kv-cache-dtype fp8_e4m3` would halve
the KV; the paper set nothing, so the recipe declares nothing.

### 3.4 Family `qwen3.8-27b` — the T4 identity/outage judge

**Shared = the single variant.** Client block from the scenarios (`identity.yaml:49-58`, `outage.yaml:52-61`):
`temperature: null`, `max_output_tokens: 16384`, `context_tokens: 131072`,
`tokenizer: Qwen/Qwen3.8-27B-FP8@017b9c7a…`, `concurrency: 64` (scenario slot value; open question 4),
`decoding: json_schema`, `image_processor: qwen3_vl`, `max_images: 10`.

| Fact | Value | Source |
|---|---|---|
| Hub repo | `Qwen/Qwen3.8-27B-FP8` | scenarios identity.yaml:28, outage.yaml:31 |
| Revision (pin) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` — pinned in the scenarios (checked 2026-10-07 there), verified == Hub head 2026-10-09 | scenarios; Hub API |
| Licence / gated | Apache-2.0 / not gated | card + Hub API |
| `config.json` architectures | `Qwen3_5ForConditionalGeneration` (same shape as qwen3.6-27b: 64 layers, hidden 5120, head_dim 256, 4 KV heads, vocab 248 320, mpe 262 144, FP8 [128,128]; same vision tower) | `config.json` at the pin |
| vLLM registry line | `Qwen3_5ForConditionalGeneration` → `registry.py:596` | tag v0.31.0 |
| Weights | 30.9 GB (24.7 GB FP8 + 6.2 GB BF16) | Hub API 2026-10-09 |

**Serve block** (the scenarios' own vLLM command, `identity.yaml:29-48` — the served name there is the slot name
`judge`; the recipe carries the catalog id, open question 2):

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 10}'
```

`--max-model-len 131072` is the scenario's choice (the checkpoint's own mpe is 262 144; it matches the client's
`context_tokens: 131072` — open question 5). No `--quantization` (FP8 auto-detected); no `--kv-cache-dtype` (auto
→ BF16).

**resources.gpus: 1** (scenario slots: `gpus: 1`, `identity.yaml:59`, `outage.yaml`). Arithmetic as qwen3.6-27b:
30.9 GB weights, 65 536 B/token KV (BF16), ≈ 151 MB per-sequence linear state; the 131 072 context on 80 GB gives
≈ 43 GB of cache; after the states of 64 concurrent sequences (≈ 9.7 GB) that is ≈ 33 GB — ≈ 250 typical 2k-token
tournament windows or ≈ 4 full-length 131 072-token sequences.

### 3.5 Family `qwen3.8-flash-next` — the T4 four-phase and ViDoRe judge (two quantisations)

**Shared (family.yaml):** served name pattern `<variant id>` (the scenarios serve the slot name `judge`, open
question 2); client block: `temperature: null`, `max_output_tokens: 16384`, `context_tokens: 131072`,
`concurrency: 256` (text run; the visual run's slot used 128 — open question 4), `decoding: json_schema`,
`image_processor: qwen3_vl`, `max_images: 10`; serve block shared except the quantisation flag:
`--tensor-parallel-size 4`, `--max-model-len 131072`, `--reasoning-parser qwen3`,
`--limit-mm-per-prompt '{"image": 10}'`. Shared checkpoint facts: `Qwen4ExpForConditionalGeneration`
(`config.json` text config: 48 layers, hidden 2560, head_dim 256, 2 KV heads, 512 experts, top-10, moe intermediate
640, vocab 248 320, mpe 262 144, `full_attention_interval` 4; vision tower as the other Qwen3.8 checkpoints);
identical `tokenizer_config.json` + chat template (sha256 prefix `b11349aafa7cdc6a`) and preprocessor geometry
(patch 16, merge 2, shortest 65 536 / longest 16 777 216 px — the product's `qwen3_vl` processor geometry,
`rcp_ndcg.data.resolution` PROCESSORS).

**Per-variant facts:**

| Fact | `qwen3.8-flash-next-nvfp4` | `qwen3.8-flash-next-fp8` |
|---|---|---|
| Hub repo | `nvidia/Qwen3.8-Flash-Next-NVFP4` | `Qwen/Qwen3.8-Flash-Next-FP8` |
| Revision (pin) | `fc694b54fb0174e0913e6adf86691ef85a4ead47` — scenarios (checked 2026-10-07), == Hub head 2026-10-09 | `236dfdf285828023ca3bcd3f37366c58a3469b13` — scenarios (checked 2026-10-07), == Hub head 2026-10-09 |
| Licence | other → `nvidia-open-model-license` (the card's governing terms; the base model is under Qwen Community 1.0) | other → `qwen-community-1.0` |
| Gated | not gated | not gated |
| Quantisation | ModelOpt NVFP4 (`config.json` `quant_method: modelopt`; `hf_quant_config.json` `quant_algo: MIXED_PRECISION` — linear-attention, hyper-connection, gate, embed and lm_head modules excluded; `kv_cache_quant_algo: null`) | FP8 block [128,128], dynamic activations (`config.json` `quant_method: fp8`) |
| Serve `--quantization` | `modelopt_fp4` (declared; detection would also map it, `modelopt.py:138`) | none (FP8 auto-detected, `fp8.py:92,138-139`) |
| `config.json` architectures | `Qwen4ExpForConditionalGeneration` | `Qwen4ExpForConditionalGeneration` |
| vLLM registry line | `registry.py:601-604` → `vllm/models/qwen4_exp` (NVIDIA implementation `vllm/models/qwen4_exp/nvidia/model.py:886`) | same |
| Weights | 132.7 GB on disk (the index's 11 shards: 60.4 GB packed NVFP4-as-U8 + 53.7 GB FP8 + 11.0 GB BF16 + FP8 scales) | 185.5 GB on disk (the index's 131 shards: 174.5 GB FP8 + 11.0 GB BF16 + FP8 scales) |
| Weights per GPU at TP4 | 33.2 GB | 46.4 GB |
| GPU class | SM100 (B200) for native FP4 GEMM; otherwise Marlin W4A16 fallback (`docs/features/quantization/modelopt.md:24-40`) | 80 GB class works (73.6 GB budget at 0.92 − 46.4 GB weights ≈ 27 GB cache); B200 comfortable |
| `resources.gpus` | 4 (scenario slots: `gpus: 4`) | 4 |
| Client tokenizer | `nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54…` | `Qwen/Qwen3.8-Flash-Next-FP8@236dfdf2…` |

Serve blocks (the scenarios' candidate and fallback commands, `text-four-phases.yaml:32-75`,
`vidore.yaml:31-74`):

```bash
# nvfp4 (the scenarios' candidate)
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --served-model-name qwen3.8-flash-next-nvfp4 \
  --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 10}'

# fp8 (the scenarios' fallback: same flags, no --quantization)
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --served-model-name qwen3.8-flash-next-fp8 \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 10}'
```

KV arithmetic (identical for both variants; both serve BF16 KV — no `--kv-cache-dtype` anywhere): 12 full-attention
layers × 2 × 2 KV heads × 256 × 2 B = 24 576 B/token across ranks; at TP4 the 2 KV heads replicate ×2 (1 head per
GPU), so 12 288 B/token per GPU (a full-length 131 072-token sequence ≈ 1.5 GiB per GPU). Linear-attention
recurrent state: 36 linear layers × 48 value heads × 128 × 128 × 4 B ≈ 113 MB per sequence, 28 MB per GPU at TP4;
at the text run's concurrency 256 that is ≈ 7.2 GB per GPU alongside the KV.

---

## 4. Engine support check (record, per workstream 08)

All six architectures are present in `vllm/model_executor/models/registry.py` at tag v0.31.0
(commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`), re-checked at the pinned revisions on 2026-10-09 (the pins'
`config.json` `architectures` are what the table shows):

| Architecture | Registry line | Serves |
|---|---|---|
| `GptOssForCausalLM` | `registry.py:123` | gpt-oss-120b |
| `Qwen3_5ForConditionalGeneration` | `registry.py:596` | qwen3.6-27b-fp8, qwen3.8-27b-fp8 |
| `Qwen3_5MoeForConditionalGeneration` | `registry.py:597-600` | qwen3.5-397b-a17b-nvfp4 |
| `Qwen4ExpForConditionalGeneration` | `registry.py:601-604` | qwen3.8-flash-next-fp8, qwen3.8-flash-next-nvfp4 |

The text-only causal variants (`Qwen3_5ForCausalLM`/`Qwen3_5MoeForCausalLM` at `registry.py:203-204`,
`Qwen4ExpForCausalLM` at `:113-116`) and the MTP heads (`Qwen4ExpMTP` at `:697`, `Qwen3_5MTP`/`Qwen3_5MoeMTP` at
`:699-700`) are also registered. Every judge therefore serves on the stock `vllm/vllm-openai:v0.31.0` image; no
judge needs a newer vLLM. Quantisation support: `modelopt_fp4` (`modelopt.py:715-746`), `fp8` (`fp8.py:92,138-139`),
`gpt_oss_mxfp4` (`models/config.py:414-421`) are all in the v0.31.0 tree; the two NVFP4 checkpoints want SM100-class
GPUs for native FP4 GEMM (Hopper works through the Marlin W4A16 fallback with a warning,
`docs/features/quantization/modelopt.md:24-40`) — a capacity note, not a blocker.

## 5. Where the client-side facts come from (R30)

The client blocks above are rcp-ndcg's judge config fields, not a second model: `context_tokens` and `tokenizer`
drive the text budget (`rcp_ndcg.judging.judging.window_tokens` — documents go whole when `context_tokens` is
unset), `decoding: json_schema` selects the structured-output path, and `image_processor: qwen3_vl` selects the
product's media geometry (`rcp_ndcg.data.resolution.PROCESSORS["qwen3_vl"]`: factor 32, image 65 536..16 777 216 px,
video per clip 4 096..25 165 824 px, one timestamp line per temporal group) — which matches all five Qwen
checkpoints' own `preprocessor_config.json` (patch 16 × merge 2, size shortest 65 536 / longest 16 777 216),
verified at the pinned revisions on 2026-10-09. The recipe's client block is that config, never a copy (decision
15, R30).

---

## 6. Open questions (for the owner / lane l08-judges)

1. **qwen3.6-27b-fp8's engine settings**: the paper recorded only the served name and the reasoning parser
   (`experiments/paper/README.md:40-42`); its checkpoint revision is not recorded (the preset carries
   `revision: null`). This spec assumes TP1 (27B-class fits an 80 GB GPU), `--max-model-len 262144` (the
   checkpoint's own limit) and today's Hub head as the pin. The recipe can also carry `revision: null` like the
   preset; a pin is recommended so the conformance replay stays reproducible.
2. **Served names**: the paper served `qwen3.5-397b` and `gpt-oss-120b` (sglang.sh:14); the T4 scenarios serve every
   judge under the slot name `judge`. The recipe's `serve.served_model_name` should be the catalog id (so
   `--judge <id>` and the client's `model` agree), with scenarios overriding per run.
3. **`max_images` for qwen3.5-397b-nvfp4 and qwen3.6-27b-fp8**: today's presets leave it unset (run configs set
   it); the recipe's client block needs it for media corpora. Recommend 10, matching the scenarios
   (`identity.yaml:57`, `text-four-phases.yaml:84`, `vidore.yaml:83`).
4. **`concurrency` values**: the presets carry 256; the scenarios carry slot-specific values (64 for
   qwen3.8-27b-fp8 on one GPU, 128 for the visual flash-next slot). One recipe value (256) with per-run overrides,
   or a per-variant override in the family table — recommend 256 as the recipe value, scenario values as runtime
   tuning.
5. **qwen3.8-27b-fp8's context length**: the scenarios run `--max-model-len 131072` (and the client
   `context_tokens: 131072`) although the checkpoint's own mpe is 262 144. Keep 131 072 (matches the client budget
   and the T4 smoke) or raise both to 262 144.
6. **KV cache dtype**: only the paper's primary judge declares FP8 KV (its `hf_quant_config.json` asks for it); the
   other five run BF16 KV by default. Halving the 27B pair's KV with `--kv-cache-dtype fp8_e4m3` is tempting for
   density, but the paper's runs used the default — the recipe declares nothing unless the owner wants the change.
7. **Family shape**: this spec keeps `qwen3.6-27b` and `qwen3.8-27b` as two single-variant families (chat template
   and generation differ; section 1) rather than one `qwen3-27b` family with two variants, and keeps the three
   Qwen3.8 checkpoints in two families (27B vs Flash-Next) even though their tokenizers are byte-identical, because
   their architectures (`Qwen3_5…` vs `Qwen4Exp…`) and shapes differ. If the owner prefers one `qwen3.8` family with
   three variants, the blocks above separate cleanly into shared + per-variant rows with no fact changes.
8. **MTP heads**: all five Qwen checkpoints ship one MTP layer (`mtp_num_hidden_layers: 1`) and vLLM v0.31.0
   registers the heads (`registry.py:697,699-700`). The paper ran no speculative decoding; the recipes don't enable
   it. Enabling it later is a deliberate, tested change (throughput only, but it changes the serve block).
