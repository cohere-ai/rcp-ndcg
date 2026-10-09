# Workstream 08: vLLM only, and recipes for every role (judges included)

Read `handover/00-MASTER.md` first (decisions 14-17, 18-19, 24). Run after workstream 05 (the layout move), so the
recipes are package data of `rcp-ndcg-vllm` and `recipe:<id>` resolution exists. One builder, two verifiers.

## A. vLLM only (decision 14)
Remove every explicit SGLang path. At M3 these files mention SGLang (`git grep -il sglang`): the rerank and embedding
adapters (e.g. the SGLang bare-list rerank parsing), `data/resolution.py` and `data/prepare.py` (SGLang media-geometry
branches), `inference/adapters/chat.py`, `llm/client.py`, `llm/_parsing/schema.py`, `llm/judges/*`, `support/serve.py`,
`retrieval/config.py`, `experiments/paper/serve/*.sglang.sh`, the docs (`docs/concepts/{judges,runs,embeddings,
preprocessing,tournament}.md`, `docs/api/inference.md`), the JSON schemas, the NOTICE rows for SGLang-derived test
oracles (`tests/data/_media_reference.py`), and the tests that exercise those branches. Remove code, tests, schema text
and NOTICE rows together (failing test first where behaviour changes: an SGLang-shaped reply is now refused by name).
`REPRODUCIBILITY.md` says: the paper's judges ran on SGLang (the submission code is the record); this release serves
them on vLLM v0.31.0.

## B. Recipes for every role (decisions 15, 18, 19)
- The recipe schema accepts `role: judge` with `client.api: chat`; a judge recipe has no `reference`; its client block is
  rcp-ndcg's judge config (validated by rcp-ndcg, R30 — never a duplicated model). The serve block carries the vLLM
  flags (tensor parallel size, reasoning parser, context length, quantization, media limits); `resources.gpus`.
- Judge validation is a T0 serve smoke plus structured-output and parse conformance (CPU-tested through the fake engine
  and, where a corpus exists, the emulators). `status: unverified`.
- `schema_version` on every recipe (decision 18), the exported JSON Schema, and rcp-ndcg's compatibility check when it
  reads `recipe:<id>` (refuse an unknown major version with the install hint).

## C. The judge catalog (cheap: public model cards plus the vLLM v0.31.0 docs and source)
`qwen3.5-397b-a17b-nvfp4`, `gpt-oss-120b`, `qwen3.6-27b-fp8`, `qwen3.8-27b-fp8`, `qwen3.8-flash-next-fp8`,
`qwen3.8-flash-next-nvfp4`. Pin each to a Hub revision (the three Qwen3.8 pins are verified in
`rcp-ndcg-vllm/scenarios/*.yaml`; resolve the others through the public Hub API and record the check date).
Client blocks come from today's presets in `rcp-ndcg/src/rcp_ndcg/llm/judges/` (context, tokenizer, decoding, image processor);
serve blocks from the model cards and the paper's engine settings (`experiments/paper/serve/`, translated to vLLM flags
— cite each flag in the vLLM v0.31.0 source).

## D. Presets become recipes
`--judge <id>` and `judge: recipe:<id>` resolve the catalog through the same path as `reranker: recipe:<id>`. Remove the
self-hosted presets in `rcp-ndcg/src/rcp_ndcg/llm/judges/*.yaml` (`qwen35_397b_fp8` is dropped: the paper never ran it); keep
`gpt5_hosted` as a vendor profile. Point the T4 scenarios, the examples and the docs at the recipes; paper configs use
`recipe:<id>` (decision 17).

## Gates
The master's quality bar; `rcp-ndcg-vllm serve <id> --dry-run` for every recipe (24); the lean install check
(`pip install --no-deps rcp-ndcg-vllm` changes `pip freeze` by exactly that wheel); `run_all` unchanged; contract
snapshots and schemas regenerated; CHANGELOG entries. Report in `handover/reports/08-vllm-only-and-judges.md`.

## Engine support check (record it in the report)
Every judge's architecture must be in vLLM v0.31.0's model registry (decision 2: no newer image). Checked at M3 against
`vllm/model_executor/models/registry.py` at the tag and the Hub `config.json` architectures:
`Qwen3_5MoeForConditionalGeneration` (qwen3.5-397b-a17b-nvfp4), `GptOssForCausalLM` (gpt-oss-120b),
`Qwen3_5ForConditionalGeneration` (qwen3.6-27b-fp8, qwen3.8-27b-fp8), `Qwen4ExpForConditionalGeneration`
(qwen3.8-flash-next-fp8/-nvfp4) — all present. Re-check at the pinned revisions you choose; a judge that needs a newer
vLLM is an owner decision, not a silent image bump.
