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
---

## 7. Gemma 4 judges (lane `judge-gemma`, 2026-10-09; four recipes, three families)

Status: research only, 2026-10-09. This section extends the catalog above with four Gemma 4 judges. It is the
deliverable of lane `judge-gemma`; the recipe files themselves are lane `l08-judges` (08 B+D). It fixes, for each
judge, every fact that lane needs in the same shape as the six judges above: the Hub repo and 40-hex revision,
licence and gated status, the checkpoint's `config.json` architectures, the vLLM v0.31.0 registry line that serves
each architecture, the ModelOpt NVFP4 path on B200 (method name, kernels, flags) and the Hopper fallback, the
reasoning parser and the guided-decoding path under the checkpoint's own chat template, the serve block with
`resources.gpus` and the memory arithmetic on an 80 GB-class GPU and on a B200, the client block from
`rcp_ndcg.judging.JudgeConfig`, the family grouping under decision 34, and the E2 GPU validation plan.

Sources and method:

- Hub facts resolved anonymously through the public Hub API (`https://huggingface.co/api/models/<repo>`, the
  repo's `config.json`, `hf_quant_config.json`, `processor_config.json`, `tokenizer_config.json`,
  `tokenizer.json`, `model.safetensors.index.json`, `chat_template.jinja` and the model card at the pinned
  revision), checked **2026-10-09**. Each pin resolves at the API to exactly the 40-hex revision in the table
  below (the API's `sha` equals the requested revision for all four).
- vLLM cloned at tag `v0.31.0` (commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`) into the lane scratch; every
  `vllm/...` citation below is `file:line` at that tag.
- transformers 5.17.0 read at the `v5.17.0` tag for the Gemma 4 image/video processor resize rules (vLLM
  v0.31.0's runtime range is `transformers >= 5.10.4, < 5.18.0`, `requirements/common.txt:10`; the tag's own
  test pins name 5.17.0, `requirements/test/cuda.txt:1303`, and the released image's exact patch version is only
  observable on the E2 node), and the checkpoint's own `chat_template.jinja` rendered with jinja2 3.1.6
  (`trim_blocks=True, lstrip_blocks=True`) for the judge-shaped prompt check in section 7.1.
- The four checkpoints are **not in the paper**: the paper's judges are the six above. Gemma 4 is an owner
  addition, so there is no paper-side engine setting to translate; the serve blocks below are this spec's
  translation of the model cards' own recommendations and the engine's defaults, and every flag is cited in the
  vLLM v0.31.0 source.
- KV arithmetic counts only the full-attention layers as per-token KV: all four checkpoints interleave
  `sliding_attention` and `full_attention` layers (`layer_types` in the text config), the full layers carry the
  larger `global_head_dim` (512) with `num_global_key_value_heads`, and `attention_k_eq_v` is true (the
  checkpoint ships a `k_proj` only on those layers; vLLM loads K into both the K and V shards, so the cache
  holds both, each `global_head_dim` wide — `gemma4.py:91-125`). The sliding layers are managed by the hybrid KV
  cache manager (HMA, enabled by default; `config/vllm.py:2222-2290`), so each holds the 1024-token window per
  sequence, not the whole context (`layers/attention/attention.py:624-657` builds a `SlidingWindowSpec`;
  `kv_cache_interface.py:487-543` is the page arithmetic: `num_kv_heads x (head_size + head_size_v) x dtype`
  bytes per token, both K and V).

---

### 7.1 The four judges at a glance and the family grouping (decision 34)

| id (recipe id) | Hub repo | Revision (pin) | Architecture | Quantisation | Licence / gated |
|---|---|---|---|---|---|
| `gemma-4-12b-it` | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | `Gemma4UnifiedForConditionalGeneration` | bf16 | apache-2.0 / not gated |
| `gemma-4-26b-a4b-it` | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `Gemma4ForConditionalGeneration` (MoE) | bf16 | apache-2.0 / not gated |
| `gemma-4-26b-a4b-nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` | `Gemma4ForConditionalGeneration` (MoE) | ModelOpt NVFP4 (experts only) | apache-2.0 / not gated |
| `gemma-4-31b-it-nvfp4` | `nvidia/Gemma-4-31B-IT-NVFP4` | `4135a98a9b728a548947683219633b25682223ac` | `Gemma4ForConditionalGeneration` (dense) | ModelOpt NVFP4 (MLP only) | other -> the card's terms are Apache-2.0 (section 7.4.3) / not gated |

| Family | Variants (recipe ids) | Why grouped / not grouped |
|---|---|---|
| `gemma-4-12b` | `gemma-4-12b-it` | Single variant. The 12B is the **unified** Gemma 4: no vision tower and no audio tower (raw patches and audio frames project straight into the LM — `gemma4_unified.py:1-31`), architecture `Gemma4UnifiedForConditionalGeneration`, and the only judge here with an `audio_config`. It shares no architecture with the other three. |
| `gemma-4-26b-a4b` | `gemma-4-26b-a4b-it`, `gemma-4-26b-a4b-nvfp4` | One model in two quantisations. Identical serving-relevant text and vision configs (30 layers, 25 sliding + 5 full, hidden 2816, 128 experts top-8, moe intermediate 704, dense MLP intermediate 2112, vocab 262 144, mpe 262 144, vision 27 layers hidden 1152 patch 16 pooling 3), identical `processor_config.json` (sha256 prefix `32bdf45d2ad4cc29`) and a **byte-identical `tokenizer.json`** (sha256 `cc8d3a0c…`; see below). The two differ in repo, revision, quantisation and the checkpoint's own chat-template file (18683 B canonical vs 16934 B NVIDIA re-export); the judge-shaped render is byte-identical (evidence below), and the differing template branches are tool-call, null-argument and multi-turn-thinking paths the judge never exercises. Declared per-variant facts: model, revision, quantisation, template sha, GPU class. |
| `gemma-4-31b` | `gemma-4-31b-it-nvfp4` | Single variant (NVFP4 only; the brief's table has no bf16 31B repo — open question 6). The 31B is dense (60 layers, 10 full, 32 query / 16 KV heads, global KV 4) and shares the architecture `Gemma4ForConditionalGeneration` with the 26B but nothing else model-specific: different depth, width, head counts, MLP shape and a different quantised target (MLP, not experts). |

**The judge-shaped render is byte-identical across all four checkpoints.** Rendered with each checkpoint's own
`chat_template.jinja` at its pin, for a system+user judge request (the shape every judge prompt has) with
`add_generation_prompt=True`, thinking off and thinking on:

- Google canonical template (12B and 26B bf16, 18683 B, sha256 prefix `ae53464bf3be2580`) vs NVIDIA re-export
  (26B NVFP4 and 31B NVFP4, 16934 B, sha256 prefix `94899c0f917d93f6`): **identical** (189 chars, thinking off;
  171 chars, thinking on).
- 12B vs 26B bf16 (both canonical): identical.

The differences between the two template files are the Google 2026-07-09 header, `preserve_thinking` handling,
the reasoning gate for tool-call turns, null-argument formatting and an O(1) continuation scan — all on paths a
single-turn judge request never reaches. The two NVIDIA templates are byte-identical to each other, and the
NVIDIA `tokenizer_config.json` lacks the Google one's `response_template` block (the Gemma response-schema
description); vLLM does not read `response_template` anywhere (no hit in the v0.31.0 tree), and the engine's
structured output comes from the request's JSON schema, not from it.

**The tokenizer is one file.** `tokenizer.json` is 32 169 626 B and byte-identical (sha256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`) in all four repos at their pins. The
post-processor adds no BOS/EOS (`post_processor` is a single sequence template), unlike the embeddinggemma-2
tokenizer; the chat template itself writes the BOS (`{{- bos_token -}}`), which is why vLLM's Gemma 4 processing
info forces `add_special_tokens=False` for chat-template models (`gemma4_mm.py:200-217`) — one BOS, from the
template. Special tokens: `<bos>` id 2, `<eos>` id 1, `<pad>` id 0, `<unk>` id 3, `<mask>` id 4; the checkpoint's
`tokenizer_config.json` names the channel tokens `soc_token`/`eoc_token`, the think trigger `think_token`, the
turn end `eot_token`, the image/audio/video placeholders at ids 258 880/258 881/258 884 and their wrappers at
258 882/258 883 (the video placeholder token is the one `extra_special_tokens` entry). The client's judgement family keys on the
`tokenizer.json` SHA-256, so the four judges' tokenizer hash is the same and the checkpoint `revision` is what
separates their families.

**Shared across the whole Gemma 4 catalog** (every family's shared block, not restated per variant below):

- `role: judge`, `client.api: chat`, no `reference` (decision 15); the client block is rcp-ndcg's judge config,
  validated by rcp-ndcg (R30).
- `decoding: json_schema`: the server constrains each answer to the stage's JSON schema (response_format
  json_schema, strict). Serve the model with its reasoning parser (`--reasoning-parser gemma4`), so the grammar
  applies to the answer, not to a thinking channel (section 7.2).
- The model card's own sampling, declared rather than defaulted: `temperature: null` (the server's default is
  the card's recommended 1.0) and `extra_body: {top_p: 0.95, top_k: 64}` (the card's "standardized sampling
  configuration across all use cases", 12B card section 1 / 26B card section 1; the 31B NVFP4 card's own
  benchmark line is `temperature=1.0, top_p=0.95`). The six judges above declare `temperature: null` too; the
  deliberate, declared instrument difference here is the `extra_body` sampling for a model that is not in the
  paper (open question 3).
- Engine: the stock image `vllm/vllm-openai:v0.31.0` only (decision 2); the recipe's engine block is
  `{name: vllm, image: "vllm/vllm-openai:v0.31.0", min_version: "0.31.0"}`. Both architectures are in vLLM
  v0.31.0's registry (section 7.2), so no judge needs a newer image.
- The recipe pins `revision` and the client's `tokenizer` at the same 40-hex commit (the `repo@revision` shape
  the catalog above uses).
- `context_tokens: 131072` and `max_model_len: 131072` (the T4 judge convention; the checkpoints' own mpe is
  262 144 — open question 4), `max_output_tokens: 16384`, `concurrency: 256` (the catalog value; the scenarios
  override per slot), `image_processor: gemma4`, `max_images: 10`, `max_videos: 0` (section 7.5).

---

### 7.2 Engine support at vLLM v0.31.0

#### 7.2.1 Registry

| Architecture | Registry line (tag v0.31.0) | Module | Serves |
|---|---|---|---|
| `Gemma4ForConditionalGeneration` | `vllm/model_executor/models/registry.py:418` | `gemma4_mm` (`vllm/model_executor/models/gemma4_mm.py`) | `gemma-4-26b-a4b-it`, `gemma-4-26b-a4b-nvfp4`, `gemma-4-31b-it-nvfp4` |
| `Gemma4UnifiedForConditionalGeneration` | `registry.py:419-421` | `gemma4_unified` (`vllm/model_executor/models/gemma4_unified.py`) | `gemma-4-12b-it` |

Also registered at the tag (not needed by the judges): the text-only `Gemma4ForCausalLM` (`registry.py:111`), the
MTP/speculator heads `Gemma4MTPModel`/`Gemma4DSparkModel` (`registry.py:658,684`) and the diffusion variant
(`registry.py:415-417`). Every judge's architecture is present; **no judge needs a newer vLLM** (decision 2).
The 12B's `Gemma4UnifiedForConditionalGeneration` subclasses the tower-based model and overrides only the
multimodal pipeline (`gemma4_unified.py:1-31`); the language model, attention, MoE and MTP paths are inherited.

Text-model facts the serve block relies on: `final_logit_softcapping` (30.0 in every text config) is applied
(`gemma4.py:1461`); the heterogeneous head dimensions (sliding 256 / full 512) are resolved per layer
(`vllm/transformers_utils/configs/gemma4.py:12-34`); `attention_k_eq_v` full layers load their `k_proj` into
both the K and V shards (`gemma4.py:91-125`); the MoE layer runs the dense MLP (the card's "1 shared" expert,
intermediate 2112) **plus** the routed experts and adds them (`gemma4.py:736-757`), with the model's own
top-k/renormalise/per-expert-scale routing kernel (`gemma4.py:130-250`).

**The 32-bit index arithmetic at 262 144.** The chosen context (131 072) and the checkpoints' own limit
(262 144) both sit far below every int32 bound in the path, checked at the tag: the input ids and the
flash-attn `cu_seqlens` are int32 (`v1/worker/gpu_model_runner.py:790`, `v1/attention/backends/flash_attn.py:272`;
position ids are int64, `:794`), the block table is int32 with `ceil(max_model_len / block_size)` blocks per
request (`v1/worker/block_table.py:115`; `kv_cache_interface.py:545-550`) — 8192 blocks at 131 072 and 16 384
at 262 144 with the default block 16 — and the Gemma 4 router packs the float32-sortable logit key and the
expert id into the two 32-bit halves of an int64 (`gemma4.py:155-165`; 128 experts, so the id half uses 7
bits). The KV pages stay small too: 16 tokens x (512 + 512) x the served shape's global KV heads = 32 KiB per
full-layer page at every served shape (12B TP1 bf16 32 KiB; 26B bf16 TP2 32 KiB per rank; 26B NVFP4 TP1 32 KiB
fp8; 31B TP2 32 KiB per rank fp8; the 31B at TP1 would be 64 KiB). No 32-bit overflow is in reach at either
context.

#### 7.2.2 The ModelOpt NVFP4 path on B200 (and the Hopper fallback)

- **Method name**: `modelopt_fp4`. The flag is `--quantization modelopt_fp4` (`engine/arg_utils.py:952`;
  `ModelOptNvFp4Config.get_name() -> "modelopt_fp4"`, `layers/quantization/modelopt.py:745-746`). vLLM also
  auto-detects it: the checkpoints' `config.json` `quantization_config` has `quant_method: "modelopt"` and
  `quant_algo: "NVFP4"`, and `ModelOptNvFp4Config.override_quantization_method` maps any `NVFP4`/`FP4` algo to
  `modelopt_fp4` (`modelopt.py:756-761`; the override loop is `config/model.py:1336-1400`). Declaring the flag
  keeps it explicit; it does not change what is loaded.
- **What is quantised**: the 26B NVFP4 quantises the **experts only** (the card's `nvfp4_experts_only` recipe):
  11 520 tensors = 30 layers x 128 experts x 3 projections carry `weight`/`weight_scale`/`weight_scale_2`/
  `input_scale`; every layer's dense `mlp`, `router` and `self_attn`, `lm_head`, `model.embed_vision*` and the
  vision tower are excluded (the `ignore` list in `config.json`, the same list as `hf_quant_config.json`'s
  `exclude_modules`). The 31B NVFP4 quantises the **dense MLP only**: 180 tensors = 60 layers x 3 projections;
  attention, `lm_head`, the vision tower and `embed_vision` are excluded.
- **W4A4, not W4A16**: both `config_groups` declare 4-bit float activations (`input_activations` non-null), so
  vLLM keeps `NVFP4` and does not route through the W4A16 downgrade (`modelopt.py:775-791`); the load path is
  the compressed-tensors-style branch (`from_config`, `modelopt.py:311-395`) fed by `config.json`'s
  `quantization_config` (`weight_utils.py:192-266`).
- **KV cache**: both checkpoints' `kv_cache_scheme` is an 8-bit float, and vLLM resolves
  `--kv-cache-dtype auto` to `fp8_e4m3` from it (`utils/torch_utils.py:448-500` and `:503-521`; the map
  `:67-70` maps `"fp8"` to `fp8_e4m3`). The checkpoints ship **no** `k_scale`/`v_scale` tensors (the index has
  none), so the scales are computed at load. The serve blocks below declare `--kv-cache-dtype fp8_e4m3`
  explicitly — the same value `auto` would pick, now recorded rather than defaulted.
- **Kernels on B200 (SM100)**: vLLM selects the GEMM kernel at load time from the backends the platform has
  (CUTLASS, FlashInfer, Marlin, ...; `docs/features/quantization/modelopt.md:22-40`). Dense NVFP4 layers take
  the linear backend chosen by `KernelConfig.linear_backend` (default `auto`; override
  `--linear-backend`, `arg_utils.py:546,1773-1775`; values relevant to NVFP4: `cutlass`, `flashinfer_cutlass`,
  `flashinfer_cutedsl`, `flashinfer_trtllm`, `flashinfer_cudnn`, `marlin`). The 26B's expert GEMMs take
  `select_nvfp4_moe_backend` (`fused_moe/oracle/nvfp4.py:182-203`), whose auto order is FLASHINFER_TRTLLM,
  FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION
  — first supported wins, and the engine logs `Using '<backend>' NvFp4 MoE backend out of potential backends`
  (`:228-233`). On Hopper (SM90) there is no native FP4 GEMM: vLLM falls back to weight-only W4A16 via Marlin
  and logs a warning (`docs/features/quantization/modelopt.md:24-40`); the 26B NVFP4 card itself notes that on
  the vLLM it was built against "the current MoE backend is either VLLM_CUTLASS or Marlin" and that
  FlashInfer-TRTLLM needed an open PR (card, "Usage"), so the E2 wave records the chosen backend rather than
  assuming one (section 7.6).
- **The 26B NVFP4 card's TP constraint**: "Currently for this model, vllm works with TP=1 only, it does support
  EP" (card, "Usage"), because of MoE/FlashInfer sharding issues in the vLLM it names. This spec therefore
  serves the 26B NVFP4 at `--tensor-parallel-size 1` (which the memory arithmetic supports comfortably on both
  GPU classes) and treats any TP>1 attempt as an E2 probe, not the default (open question 5).

#### 7.2.3 The reasoning parser, the thinking channel and the guided-decoding path

Gemma 4 **has** a thinking channel, and vLLM v0.31.0 ships its parser:

- The parser is registered as `gemma4` (`reasoning/__init__.py:55-58` -> `gemma4_engine_reasoning_parser` ->
  `Gemma4ParserReasoningAdapter`, `parser/engine/registered_adapters.py:46-49`), implemented by `Gemma4Parser`
  (`parser/gemma4.py:409-581`) over the checkpoint's channel tokens (`CHANNEL_START`/`CHANNEL_END`,
  `parser/gemma4.py:38-39`; the same pair the `tokenizer_config.json` names `soc_token`/`eoc_token`). The
  class docstring and the transition table treat a tool call as an implicit reasoning end
  (`parser/gemma4.py:409-421`), and `extract_reasoning` strips the `thought` role label (`:570-581`).
- **Thinking is off by default in the checkpoint's own template**: `enable_thinking` defaults to false, and with
  `add_generation_prompt` the template pre-closes an **empty** channel (it writes CHANNEL_START + `thought` +
  newline + CHANNEL_END after the model turn opener). With `enable_thinking: true` the template injects the
  think token at the top of the first system turn and leaves the model turn open, so the model generates its own
  channel (checkpoint `chat_template.jinja`, the `add_generation_prompt` block; the model cards' "Thinking Mode
  Configuration": with thinking disabled the model "will still generate the tags but with an empty thought
  block" — the pre-closed prompt is what makes that a no-op).
- **Guided decoding works with this template and parser.** vLLM applies the JSON-schema grammar only after
  reasoning ends: `_get_constraint_start` returns 0 while `enable_in_reasoning` is false (default,
  `config/structured_outputs.py:41`) and the request's reasoner says reasoning has ended
  (`v1/structured_output/__init__.py:235-262`). `Gemma4Parser`'s engine config sets
  `wait_for_reasoning=thinking` with initial state CONTENT (`parser/gemma4.py:297-320`), and
  `ParserEngine.is_reasoning_end` scans the prompt backwards for a channel end / turn boundary
  (`parser/engine/parser_engine.py:650-672`). Concretely:
  - **thinking off** (the default, nothing sent): the prompt ends with the pre-closed empty channel, the scan
    hits the channel end first -> `is_reasoning_end(prompt)` is true -> the grammar constrains the **first
    generated token**. The answer is pure JSON in `content`; no channel tokens can leak (the grammar forbids
    them). The model card's "the model still generates the tags" case cannot arise under the grammar; the
    parser also handles it defensively (it injects a channel start when the first feed begins with the thought
    label or the channel end, `parser/gemma4.py:446-478`).
  - **thinking on** (`extra_body.chat_template_kwargs: {enable_thinking: true}`): the prompt ends open; the
    model emits its channel, the parser separates it into `reasoning`, and the grammar starts at the channel
    end. The `reasoning_effort` request field also enables it automatically (`protocol.py:583-586`), but the
    parser reads `chat_template_kwargs.enable_thinking` at construction (`parser/gemma4.py:423-426`), so the
    recipe declares the template kwarg itself, never `reasoning_effort`.
- **What the client must set**: `decoding: json_schema` (it does), and — because the judge's answers carry no
  separate reasoning channel when thinking is off — the client's own reasoning watch (`inference/adapters/chat.py:
  415-428`) logs its one advisory warning after 8 answers ("the server probably runs the model without its
  reasoning parser"). That warning is a false positive for a thinking-off Gemma 4 judge: the model's reasoning
  is the JSON's own `reasoning` field (the tournament and rubric prompts ask for it), not a channel. The
  recipe's `notes` should say so (open question 2).
- Structured-output support for the `gemma4` parser is documented as `json`, `regex`
  (`docs/features/reasoning_outputs.md:20`), and the parser is listed with `enable_thinking` off by default
  (`:37`). Tool calling is not needed (the judge never requests tools; the parser's tool branch is unused).

#### 7.2.4 Attention backends (the 512-wide full layers)

Gemma 4's full-attention layers have `global_head_dim` 512, which not every attention backend supports: vLLM's
FlashAttention backend accepts 512 only when FA4 is available (`v1/attention/backends/flash_attn.py:421-429`),
and the SM90 version policy upgrades FA3 to FA4 for head sizes above 256
(`v1/attention/backends/fa_utils.py:99-102`, `:145-161`). At v0.31.0 the engine prefers, per platform
(`platforms/cuda.py:154-177`):

- **Blackwell (SM100)**: `TRITON_FLASHINFER` for compatible multimodal-prefix configurations — including Gemma 4
  with BF16 or FP8 KV cache — falling back to `FLASHINFER`, `FLASH_ATTN`, `TRITON_ATTN`. The composite supports
  head dimensions 256/512, FP16/BF16 and FP8 KV, and 64-token kernel pages; for Gemma 4, full CUDA graphs cover
  single-token batches (`docs/design/attention_backends.md:77-104`).
- **Hopper (SM90)**: `TRITON_FLASH_ATTN` for multimodal-prefix configurations, with the causal child resolving
  to FA4 — which happens automatically for FA4-only shapes such as head size 512 and for Gemma 4
  (`docs/design/attention_backends.md:60-73`).

So the Gemma 4 judges are Hopper-or-Blackwell judges; an A100-80GB class node is not the target (the hd512
full layers would need a fallback backend). No `--attention-backend` override is declared: the platform picks
it, and the E2 smoke records the engine's choice.

#### 7.2.5 Media at the engine (image and video)

- **Image geometry** (all four checkpoints): the checkpoint's own image processor resizes by
  `get_aspect_ratio_preserving_size` — scale both edges by `sqrt(max_soft_tokens x pooling^2 x patch^2 / area)`,
  floor each to `pooling x patch` = 48 px, cap at the patch budget — with patch 16, pooling 3,
  `max_soft_tokens` 280 (`processor_config.json` `image_processor`; the identical function in transformers
  5.17.0 `models/gemma4/image_processing_gemma4.py:33-84` and
  `models/gemma4_unified/image_processing_gemma4_unified.py:54-105`). vLLM's `Gemma4ProcessingInfo` reproduces
  it for the prompt-side placeholder count (`gemma4_mm.py:285-323` `_compute_num_soft_tokens`, used by
  `get_image_repl` `:325-...`), and the unified variant overrides only the field it reads
  (`gemma4_unified.py:184-200`).
- **Video** is frames through the same image path: vLLM's Gemma 4 constants are 70 soft tokens per frame and at
  most 32 frames (`gemma4_mm.py:102-104`; the video processor's own defaults are `max_soft_tokens: 70`,
  `num_frames: 32`), and the checkpoint renders one wrapper pair per frame plus a per-frame timestamp line
  (transformers 5.17.0 `models/gemma4/processing_gemma4.py:179-188`: an `mm:ss` timestamp, then the frame's
  vision block; vLLM budgets `num_frames x (70 + 2 + 6)` for it, `gemma4_mm.py:271-276`, and requires the
  container's metadata, `gemma4_mm.py:278-283`). The visual-token budgets the NVIDIA cards list (70, 140, 280,
  560, 1120) are the processor's accepted set, mirrored in vLLM's `_SUPPORTED_SOFT_TOKENS` (`gemma4_mm.py:102`);
  `--mm-processor-kwargs` can pin a budget (`{"images_kwargs": {"max_soft_tokens": 280}}` or top level;
  `_get_max_soft_tokens`, `gemma4_mm.py:107-121`), but the checkpoints' own 280 is what the client prepares, so
  no pin is declared.
- **Engine limits**: `--limit-mm-per-prompt` takes the legacy count or the configurable shape
  (`config/multimodal.py:136-145`; flag `engine/arg_utils.py:1432`), and `--media-io-kwargs` pins per-modality
  processing such as `num_frames` (`config/multimodal.py:159-162`). The default video sampler samples
  `num_frames` uniformly over the clip (`multimodal/video.py:202-240`), so a pinned frame count is honoured on
  v0.31.0 (unlike the embeddinggemma-2 nightly backend the `rec-egemma2` lane found).
- The judge's vision path is page images (the paper's `vidore` scenario): `max_images: 10` on the client and
  `--limit-mm-per-prompt '{"image": 10}'` at the engine. Video is opt-in (open question 5); the NVIDIA cards
  describe it as up to 60 s at one frame per second, which is a card-side rate, not the engine's default (the
  engine samples `num_frames` uniformly unless pinned, and the client's `wire: frames` path samples on the
  client).

---

### 7.3 Flag translation for the Gemma 4 judges (one table)

The catalog-wide table above (section 2) already cites the vLLM argument definitions; this table lists only the
flags the Gemma 4 serve blocks add and the ones deliberately not declared.

| vLLM flag (value) | Why | vLLM v0.31.0 citation |
|---|---|---|
| `--tensor-parallel-size N` | per-family in section 7.4 | `arg_utils.py:1151` |
| `--max-model-len 131072` | the T4 judge context; the checkpoint allows 262 144 | `arg_utils.py:951` |
| `--reasoning-parser gemma4` | the schema applies after the thinking channel; the empty pre-closed channel is recognised | `arg_utils.py:1093`; `reasoning/__init__.py:55-58` |
| `--quantization modelopt_fp4` | NVFP4 checkpoints only; auto-detection would map it, the flag records it | `arg_utils.py:952`; `modelopt.py:745-746` |
| `--kv-cache-dtype fp8_e4m3` | NVFP4 checkpoints only: the checkpoint's `kv_cache_scheme`; `auto` resolves to the same value | `arg_utils.py:1334`; `utils/torch_utils.py:67-70,503-521` |
| `--limit-mm-per-prompt '{"image": 10}'` | matches the client's `max_images`; without it each modality defaults to 999 | `arg_utils.py:1432`; `config/multimodal.py:127-145` |

Not declared, and why:

- `--dtype`: `auto` reads the checkpoints' `config.json` `dtype: bfloat16`; the NVFP4 checkpoints' unquantised
  parts are bf16 too. A `--dtype bfloat16` spelling is equivalent; the catalog's judge blocks leave it to the
  checkpoint.
- `--chat-template`: the checkpoint's own `chat_template.jinja` is the instrument (the four renders are checked
  in section 7.1); transformers loads it by name (5.17.0 `utils/hub.py:67`, `CHAT_TEMPLATE_FILE =
  "chat_template.jinja"`) and `gemma4_mm.py:200-217` then suppresses the duplicate BOS. No template file ships
  with the recipe.
- `--trust-remote-code`: the checkpoints have no `auto_map`; the architectures are native at the tag. The
  cards' sample commands pass it against much older vLLM images (the 31B card's names 0.17.2rc1; the 26B card's
  v0.20.0 already has the architecture, so its flag is boilerplate). No flag at v0.31.0.
- `--linear-backend` / `--moe-backend`: vLLM selects at load (`docs/features/quantization/modelopt.md:22-40`;
  `fused_moe/oracle/nvfp4.py:182-203`). The E2 wave records the chosen backends from the engine log; a pin is a
  deliberate change, not a default.
- `--gpu-memory-utilization`: the engine default (0.92) is the arithmetic below; the paper's `0.85` does not
  apply to judges the paper never ran.
- `--enable-auto-tool-choice` / `--tool-call-parser gemma4`: the judge never requests tool calls.
- `--media-io-kwargs`: only for a `video_url` corpus (the frame pin); the page-image path needs none.
- MTP/speculative decoding: the checkpoints' MTP heads are registered at the tag, but the paper ran no
  speculative decoding and the recipes do not enable it (as the six above).

---

### 7.4 The three families in detail

Every variant's row carries `id`, `model` (Hub repo), `revision`, and the per-variant overrides. Everything
else is the family's shared block. `resources.gpus` is the per-replica count; the B200 alternative is stated
with the arithmetic. Weight sizes are the safetensors the loader reads (the index's `metadata.total_size`, or
the single file's size for the 12B); "params" are the stored parameters at the pin.

#### 7.4.1 Family `gemma-4-12b` — the encoder-free unified judge

**Shared = the single variant.** Client block: `temperature: null`, `extra_body: {top_p: 0.95, top_k: 64}`,
`max_output_tokens: 16384`, `context_tokens: 131072`,
`tokenizer: google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, `concurrency: 256`,
`decoding: json_schema`, `image_processor: gemma4`, `max_images: 10`.

| Fact | Value | Source |
|---|---|---|
| Hub repo | `google/gemma-4-12B-it` | Hub API 2026-10-09 |
| Revision (pin) | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (the Hub head; last modified 2026-07-20) | Hub API |
| Licence / gated | apache-2.0 / not gated | card + Hub API |
| `config.json` architectures | `Gemma4UnifiedForConditionalGeneration`, model_type `gemma4_unified`; text config 48 layers (40 sliding + 8 full), hidden 3840, intermediate 15 360, 16 query / 8 KV heads, head 256 / global head 512, global KV 1, sliding window 1024, vocab 262 144, mpe 262 144, `final_logit_softcapping` 30.0, `attention_k_eq_v`, tied embeddings; `vision_config` `gemma4_unified_vision` (no tower: `model_patch_size` 48, patch 16, pooling 3, `num_soft_tokens` 280); `audio_config` non-null (the only audio judge) | `config.json` at the pin |
| vLLM registry line | `Gemma4UnifiedForConditionalGeneration` -> `registry.py:419-421` | tag v0.31.0 |
| Weights | 23 919 549 408 B in one `model.safetensors` (11 959 730 224 BF16 params, 22.28 GiB) | Hub `x-linked-size` + API |
| Processor | `Gemma4UnifiedProcessor`; `image_processor` patch 16, pooling 3, `max_soft_tokens` 280, `image_seq_length` 280; video processor 70 soft tokens, 32 frames; audio 750 tokens (unused) | `processor_config.json` at the pin |
| Chat template | canonical Google (18683 B, sha256 prefix `ae53464bf3be2580`), identical to the 26B bf16's | `chat_template.jinja` at the pin |
| Card facts | 256K context; text/image/audio (card table) and video (card feature list); encoder-free unified; thinking configurable, off by default; sampling temperature 1.0 / top_p 0.95 / top_k 64 | card |

**Serve block:**

```bash
vllm serve google/gemma-4-12B-it \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --served-model-name gemma-4-12b-it \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image": 10}'
```

**resources.gpus: 1 (TP1).** Arithmetic (GiB, `gpu_memory_utilization` 0.92 on both classes): weights
22.28 GiB (23.9 GB); KV per full layer 2 x 1 x 512 x 2 B = 2048 B/token, 8 full layers = **16 384 B/token**;
the 40 sliding layers hold their 1024-token window: 40 x 1024 x 8192 B = **0.31 GiB per sequence** (HMA). A
32k-token judge window costs 0.50 GiB of full KV + 0.31 GiB of window = 0.81 GiB; a full-length (131 072)
sequence 2.00 + 0.31 = 2.31 GiB. The 80 GB class (68.55 GiB budget) leaves 46.27 GiB of cache -> ~57
concurrent 32k windows (or ~20 full-length sequences); a 192 GB B200 (164.51 GiB) leaves 142.23 GiB -> ~175
windows. `--tensor-parallel-size 2` only halves the weights (11.14 GiB/GPU): the full layers' single global KV
head cannot be split (it replicates, so the full KV per GPU does not move), while the 8 sliding KV heads split
to 4 per rank and halve the window state to 0.16 GiB per sequence.

#### 7.4.2 Family `gemma-4-26b-a4b` — the MoE judge in bf16 and NVFP4

**Shared (family block):** served name = the variant id; client block: `temperature: null`,
`extra_body: {top_p: 0.95, top_k: 64}`, `max_output_tokens: 16384`, `context_tokens: 131072`,
`concurrency: 256`, `decoding: json_schema`, `image_processor: gemma4`, `max_images: 10`; serve block shared
except the quantisation flags and TP (below). Shared checkpoint facts: `Gemma4ForConditionalGeneration`
(text config 30 layers = 25 sliding + 5 full, hidden 2816, dense MLP intermediate 2112 — the card's "1 shared"
expert — 128 experts top-8, moe intermediate 704, 16 query / 8 KV heads, head 256 / global head 512, global KV
2, sliding window 1024, vocab 262 144, mpe 262 144; vision `gemma4_vision`, 27 layers, hidden 1152, patch 16,
pooling 3, `default_output_length` 280, position embedding 10 240, `standardize`); identical
`processor_config.json` (sha256 prefix `32bdf45d2ad4cc29`); byte-identical `tokenizer.json`. Card facts (both
cards): 256K context, text/image, video as frames, MoE 8 active of 128 experts, vision encoder ~550M params;
the NVIDIA card adds "ready for commercial/non-commercial use".

| Fact | `gemma-4-26b-a4b-it` | `gemma-4-26b-a4b-nvfp4` |
|---|---|---|
| Hub repo | `google/gemma-4-26B-A4B-it` | `nvidia/Gemma-4-26B-A4B-NVFP4` |
| Revision (pin) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (Hub head, 2026-07-20) | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` (Hub head, 2026-05-11) |
| Licence | apache-2.0 | apache-2.0 |
| Quantisation | bf16 (25 805 936 206 BF16 params, 51 611 872 412 B, 2 shards) | ModelOpt NVFP4, experts only: 11 418 992 640 U8 + 1 427 374 080 F8 scales + 2 967 950 926 BF16 params; 18 782 360 732 B on disk (the bytes add the 92 160 B of F32 scale scalars), 2 shards; 11 520 quantised tensors (30 x 128 x 3) |
| Client tokenizer | `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| `quantization_config` | absent | `quant_algo NVFP4`, `config_groups` 4-bit float weights **and** activations group 16, `kv_cache_scheme` 8-bit float, `ignore` = dense mlp/router/self_attn of every layer + `lm_head` + `embed_vision*` + `vision_tower*`, producer `modelopt 0.43.0rc2.dev91+gc79ebc014`; `hf_quant_config.json` (legacy schema) says the same (`quant_algo NVFP4`, `kv_cache_quant_algo FP8`, `group_size 16`, the same `exclude_modules`) |
| Chat template | canonical Google (18683 B, `ae53464bf3be2580`) | NVIDIA re-export (16934 B, `94899c0f917d93f6`); judge-shaped render byte-identical to the canonical one (section 7.1) |
| Card engine notes | no vLLM command on the card | "vllm works with TP=1 only, it does support EP"; "the current MoE backend is either VLLM_CUTLASS or Marlin"; test hardware B200; card claims modelopt 0.43.0 |

**Serve blocks:**

```bash
# gemma-4-26b-a4b-it (bf16)
vllm serve google/gemma-4-26B-A4B-it \
  --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
  --served-model-name gemma-4-26b-a4b-it \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image": 10}'

# gemma-4-26b-a4b-nvfp4
vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --revision a19cfe00be84568a6867111c9a68c9c44fdcffe6 \
  --served-model-name gemma-4-26b-a4b-nvfp4 \
  --tensor-parallel-size 1 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 131072 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image": 10}'
```

**resources.gpus:** `gemma-4-26b-a4b-it`: **2 (TP2)**; `gemma-4-26b-a4b-nvfp4`: **1 (TP1)**. Arithmetic:

- bf16, TP2 (GiB, 0.92 on both classes): weights 24.03 GiB/GPU; KV per GPU: 5 full layers x 2 x (2 KV heads /
  2) x 512 x 2 B = **10 240 B/token**; sliding: 25 x 1024 x 2 x 4 x 256 x 2 B = **0.10 GiB per sequence**. A
  32k window costs 0.31 + 0.10 = 0.41 GiB; the 80 GB class leaves 44.51 GiB -> ~108 concurrent 32k windows.
  A B200 leaves 140.48 GiB -> ~343 windows; TP1 (48.07 GiB weights) leaves 116.44 GiB -> ~142 windows.
- NVFP4, TP1: weights 17.49 GiB; KV (fp8) per full layer 2 x 2 x 512 x 1 = 2048 B, 5 layers = **10 240 B/token**;
  sliding 25 x 1024 x 2 x 8 x 256 x 1 = **0.10 GiB per sequence**. The 80 GB class leaves 51.05 GiB -> ~124
  32k windows; a B200 leaves 147.02 GiB -> ~358 windows. TP stays 1 per the card's constraint; the E2 probe
  may try TP2/EP as an experiment only.

#### 7.4.3 Family `gemma-4-31b` — the dense NVFP4 judge

**Shared = the single variant.** Client block: `temperature: null`, `extra_body: {top_p: 0.95, top_k: 64}`,
`max_output_tokens: 16384`, `context_tokens: 131072`,
`tokenizer: nvidia/Gemma-4-31B-IT-NVFP4@4135a98a9b728a548947683219633b25682223ac`, `concurrency: 256`,
`decoding: json_schema`, `image_processor: gemma4`, `max_images: 10`.

| Fact | Value | Source |
|---|---|---|
| Hub repo | `nvidia/Gemma-4-31B-IT-NVFP4` | Hub API 2026-10-09 |
| Revision (pin) | `4135a98a9b728a548947683219633b25682223ac` (Hub head, last modified 2026-07-13) | Hub API |
| Licence / gated | Hub tag `other`, `license_name: apache-license-2.0`, `license_link: https://ai.google.dev/gemma/apache_2`; the card states Apache License 2.0 and "This model is ready for commercial/non-commercial use", reports its own benchmark results (GPQA Diamond, AIME 2025, MMLU Pro, LiveCodeBench, Scicode, Terminal-Bench Hard) and asks users to have rights to their input media. Apache-2.0 permits use, reproduction, modification, redistribution of the weights and configs and derivative works, with attribution and the licence text, and benchmark/evaluation use; the Hub's `other` is the YAML's `license: other` + `license_name` spelling, not a different grant. Not gated. | card + Hub API |
| `config.json` architectures | `Gemma4ForConditionalGeneration`; text config 60 layers (50 sliding + 10 full), hidden 5376, intermediate 21 504, 32 query / 16 KV heads, head 256 / global head 512, global KV 4, sliding window 1024, vocab 262 144, mpe 262 144; vision as the 26B's; `quantization_config` present | `config.json` at the pin |
| Quantisation | ModelOpt NVFP4, dense MLP only: 180 quantised tensors (60 x 3), 10 404 495 360 U8 + 10 464 098 156 BF16 params and 1 300 561 920 B of F8 scale tensors; 32 633 255 032 B on disk, 4 shards (the Hub API's parameter metadata omits the F8 scales for this repo; the safetensors headers carry them); `ignore` = `lm_head`, every layer's `self_attn*`, `embed_vision*`, `vision_tower*`; `kv_cache_scheme` 8-bit float; producer `modelopt 0.37.0` in `config.json` (the card claims 0.42.0; the pinned file is the record) | `config.json` + `hf_quant_config.json` + index + safetensors headers at the pin |
| vLLM registry line | `Gemma4ForConditionalGeneration` -> `registry.py:418` | tag v0.31.0 |
| Chat template | NVIDIA re-export (16934 B, `94899c0f917d93f6`), byte-identical to the 26B NVFP4's | `chat_template.jinja` at the pin |
| Card engine notes | quantised with `nvidia-modelopt`; test hardware H100; "Supported Hardware Microarchitecture Compatibility: NVIDIA Blackwell"; sample command `--quantization modelopt --tensor-parallel-size 8` targets an older vLLM (0.17.2rc1) — at v0.31.0 the method name is `modelopt_fp4` | card |

**Serve block:**

```bash
vllm serve nvidia/Gemma-4-31B-IT-NVFP4 \
  --revision 4135a98a9b728a548947683219633b25682223ac \
  --served-model-name gemma-4-31b-it-nvfp4 \
  --tensor-parallel-size 2 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 131072 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image": 10}'
```

**resources.gpus: 2 (TP2).** Arithmetic (GiB, 0.92 on both classes): weights 15.20 GiB/GPU; KV (fp8) per
GPU: 10 full layers x 2 x (4 KV heads / 2) x 512 x 1 B = **20 480 B/token**; sliding: 50 x 1024 x 2 x 8 x
256 x 1 B = **0.20 GiB per sequence**. A 32k window costs 0.63 + 0.20 = 0.82 GiB; the 80 GB class leaves
53.35 GiB -> ~65 concurrent 32k windows. On a B200, TP2 leaves 149.31 GiB -> ~182 windows; TP1 (30.39 GiB
weights) leaves 134.12 GiB and costs 1.25 + 0.39 = 1.64 GiB per 32k window -> ~82 windows, so TP2 is the
better B200 shape here. The card's TP8 is a Hopper-era sample, not a requirement: the dense layers shard
cleanly.

---

### 7.5 The client block and the media geometry (R30)

The client blocks above are `rcp_ndcg.judging.JudgeConfig` fields, not a second model (decision 15, R30):
`context_tokens` and `tokenizer` drive the text budget (`rcp_ndcg.judging.judging.window_tokens`), `decoding:
json_schema` selects the structured-output path, `extra_body` carries the card's sampling, and
`image_processor` selects the product's media geometry (`rcp_ndcg.data.resolution.PROCESSORS`).

**The Gemma 4 image geometry is the same tower family as embeddinggemma-2's, and the `rec-egemma2` geometry is
reusable.** `rcp_ndcg.data.resolution` gains `"gemma4"` in lane `rec-egemma2` (`ImageProcessor` literal,
`PROCESSORS["gemma4"]` with `resize="gemma4"`, `soft_tokens=280`, `video_soft_tokens=140`,
`per_frame_wrapper=True`, factor 48; `gemma4_resize` / `gemma4_fixed_point` are faithful ports of
`get_aspect_ratio_preserving_size`). Checked against the four checkpoints at their pins:

- **Image: exact match.** Every checkpoint's image processor is patch 16, pooling 3, `max_soft_tokens` 280,
  `image_seq_length` 280, and the resize function is byte-identical between the tower (`gemma4`) and unified
  (`gemma4_unified`) transformers modules at 5.17.0. `image_processor: gemma4` with the geometry's 280 budget
  reproduces the engine's own resize, so the client's prepared image is the fixed point the engine keeps
  (`gemma4_fixed_point`); no `mm_processor_kwargs` pin is needed. The vision judge path (page images) is
  therefore a drop-in reuse of the `rec-egemma2` entry — **provided that lane's product change is merged before
  the judge recipes load** (it is not on `rfc-0001` today: the current `ImageProcessor` literal has only the
  three Qwen families). If it is not merged, `image_processor: gemma4` fails JudgeConfig validation and the
  recipes cannot be built (a hard dependency, not a fallback: a native policy sends images unchanged and the
  judge's media tokens cannot be counted).
- **Video: the budgets differ, and that is fine for the judge's wire.** embeddinggemma-2's video processor
  targets 140 soft tokens per frame; the four chat checkpoints' video processors target **70** per frame with
  32 frames (`processor_config.json` `video_processor`, transformers 5.17.0 `video_processing_gemma4.py:165-168`),
  and a container's prompt charges a per-frame timestamp line on top (section 7.2.5). The judge's video path is
  `wire: frames` (the client samples the frames and sends each as its own image, sized by the **image** policy
  at 280 soft tokens), so `PROCESSORS["gemma4"]`'s `video_soft_tokens` and its timestamp charge are never read
  for it. A `wire: video_url` corpus would read both and be wrong (140 vs 70 per frame, and no timestamp
  charge): that needs a per-checkpoint video budget and timestamp accounting in the product, or `wire: frames`
  (open question 5).
- **`max_images: 10`, `max_videos: 0`**: the T4 scenarios' judge value (10) and the engine's
  `--limit-mm-per-prompt` matching it. The 12B's audio is left at the engine's default and never sent: the
  product's content model has no audio part.

---

### 7.6 GPU validation plan for E2 (serve smoke, structured-output and parse conformance)

Per judge (four recipes, three families), on the E2 judge wave, with `status: unverified` until it passes
(decision 1). The CPU side first (l08-judges): the recipe contract tests run stage 1 on the real tokenizer
(network-gated) and the fake judge; the client block is validated by `rcp_ndcg.judging.JudgeConfig` (R30), and
`rcp-ndcg-vllm serve <id> --dry-run` renders the argv.

1. **T0 serve smoke** (per recipe, `serve_argv(recipe)` on the node; `rcp_ndcg_test.e2e.t0_smoke`): the engine
   boots within `startup_timeout_s`; `GET /v1/models` lists the served id with `max_model_len` 131072; one
   short chat completion returns 2xx. Record in `status.json`: the engine version, the served model entry, the
   dtype from the engine's startup log (the `/v1/models` entry has no dtype field at v0.31.0,
   `vllm/entrypoints/serve/engine/protocol.py:105-113`), the answer's `usage` and `finish_reason`, and — for
   the two NVFP4 variants — the
   engine log's chosen linear and MoE backends (`Using '<backend>' NvFp4 MoE backend out of potential
   backends`, `fused_moe/oracle/nvfp4.py:228-233`) and any Marlin W4A16 fallback warning
   (`docs/features/quantization/modelopt.md:24-40`). The 26B NVFP4 runs TP1 (the card's constraint); a TP2
   attempt is a separate probe, not the default.
2. **Structured-output conformance** (per judge): send the tournament stage's schema and the rubric stage's
   schema as `response_format: {type: json_schema, json_schema: {name, schema, strict: true}}` on a two-document
   text window (and, for the vision path, a one-page-image window through the vision prompt). Pass: 2xx; the
   answer's `content` is JSON that validates against the same schema the client sends
   (`rcp_ndcg.judging._parsing`); no channel tokens appear in `content`. Repeat once with
   `chat_template_kwargs: {enable_thinking: true}`: pass when the channel text comes back in the reasoning
   field and the `content` still validates (the parser strips the channel; the grammar starts at its end).
3. **Parse conformance** (per judge): feed the engine's answers through the judge client's own parser and store
   the judgement; pass when a parsed answer is recorded (no invalid judgement, no re-ask), for both stages and
   both thinking modes. The thinking-off run is expected to carry no separate reasoning channel — the client's
   reasoning watch logs its one warning; the run records it and continues (open question 2).
4. **Media probe** (per judge, the vision path): one page image through the vision prompt; pass when the
   request is 2xx, the image limit is honoured, and the engine's media token count equals the client's
   (`engine_media_check`, the M-media stage); the 12B additionally proves the encoder-free unified image path
   (`gemma4_unified.py:348-407`), the 26B/31B the tower path.
5. **MoE probe** (26B variants): one text request long enough to exercise all 30 layers; the bf16 variant is
   the control for the NVFP4 one (the same request, both engines, both 2xx and both parsing). This is the probe
   that would catch a ModelOpt MoE backend or weight-name problem; the checkpoint's expert naming
   (`experts.{i}.{gate,up,down}_proj.weight` + `weight_scale`/`weight_scale_2`/`input_scale`) is what the
   loader must map.
6. **E2E**: run T4 scenario 1 (text, four phases) and scenario 3 (identity) with the judge recipe in the judge
   slot, and scenario 4 (visual documents) with the vision judge; pass when the phases complete, the judgement
   store holds parsed judgements and the identity rerun recomputes nothing. Then the recipe's `status` moves to
   `verified` per `handover/RELEASE-CHECKLIST.md`.

The wave records, per judge: the chosen backends (NVFP4), the media count comparison, the two structured-output
modes, and the sustainable concurrency observed against the arithmetic in section 7.4 (the client's 256 is a
ceiling, not a promise).

---

### 7.7 Open questions (for the owner / lane l08-judges)

1. **The 26B family shape.** The bf16 and NVFP4 checkpoints' chat-template files and `tokenizer_config.json`
   differ (Google canonical vs NVIDIA re-export; the NVIDIA one lacks `response_template`), but the
   judge-shaped render is byte-identical for both thinking modes and vLLM reads neither difference. This spec
   groups them (decision 34's rule read as "materially different for this instrument"). If the owner prefers
   strict file identity, the blocks split cleanly into `gemma-4-26b-a4b` (bf16) and `gemma-4-26b-a4b-nvfp4`
   with no fact changes.
2. **Thinking off or on.** This spec keeps the checkpoint default (off): the answer is direct schema-constrained
   JSON, the prompt's own `reasoning` field carries the self-check, and the only cost is the client's one
   advisory reasoning-watch warning. The alternative is
   `extra_body: {chat_template_kwargs: {enable_thinking: true}}` (the channel is parsed and the schema applies
   after it; more tokens, and a different instrument). Recommend off; the owner may want the reasoning channel
   for judge-quality parity with the Qwen judges.
3. **Sampling.** The cards recommend temperature 1.0 / top_p 0.95 / top_k 64 (and the 31B card's own benchmark
   used temperature 1.0 / top_p 0.95). This spec declares `temperature: null` (server default 1.0) +
   `extra_body: {top_p: 0.95, top_k: 64}`; the six judges above send nothing and take the engine defaults. Keep
   the card's sampling (recommended) or match the catalog's convention.
4. **Context.** 131 072 chosen for all four (the T4 judge convention). The checkpoints allow 262 144; the
   arithmetic in section 7.4 covers it on a B200 (the sliding-window state scales with concurrency, not
   context). Raise both `max_model_len` and `context_tokens` together if a rubric pass over very many documents
   needs it.
5. **Video.** The page-image path is the judge's media path. For video corpora, `wire: frames` works with the
   reused `gemma4` image geometry; `wire: video_url` needs the product's `PROCESSORS["gemma4"]` video budget to
   be per-checkpoint (70 here vs embeddinggemma-2's 140) and the recipe to pin
   `--media-io-kwargs '{"video": {"num_frames": 32}}'`. Recommend `wire: frames` for 0.0.1 and a product note.
6. **The 31B bf16 sibling.** The catalog carries NVFP4 only (the brief's table). If the owner wants a bf16 31B
   (`google/gemma-4-31B-it`), it would be a third variant of the `gemma-4-31b` family; its arithmetic is the
   same as the 26B bf16's class (~62 GB weights, TP2 on 80 GB).
7. **The 26B NVFP4 card's producer/back-end notes.** The card says modelopt 0.43.0 and "TP=1 only" / "VLLM_CUTLASS
   or Marlin", written against an older vLLM; the pinned `config.json` says 0.43.0rc2.dev91+gc79ebc014, and
   v0.31.0's oracle may pick another backend. The E2 wave records what actually loads (section 7.6); if the
   oracle picks a backend the card calls unsupported, that is a finding, not a spec change.
8. **Audio on the 12B.** The 12B accepts audio at the engine; the judge path never sends it (`max_videos: 0`,
   no audio part in the product's content model). Out of scope for 0.0.1, as for embeddinggemma-2.
