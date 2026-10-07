<!-- Handover copy of the operator's working note `drafts/recipe-sweep-after-p1-tail.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Recipe sweep (after p1-tail and the harness merge; one lane or follow-ups per recipe)
Adopt the product features p1-tail adds, uniformly across recipes, and delete the workarounds:
- `normalize: [strip]` (and `lowercase` where the checkpoint lowercases) on the template shapes instead of jinja
  `| trim` in a template file or a declared whitespace divergence: zerank-1-reranker (template trims),
  lowercase).
- per-shape `query_max_tokens` for embed/multi_vector roles where the reference caps queries (topk 1024; check
  jina-embeddings-v5, qwen3-vl-embedding).
- the empty `{fixed: ""}` tail markers removed once the harness audit fix (FOLLOWUP-2 item 10) lands: octen,
  qwen3-embedding-0.6b, and every recipe the harness's sweep lists.
- re-run stage 1 for each changed recipe; fold every recipe's CHANGELOG bullets into one entry.
- every recipe ships `requirements-reference.txt` (qwen3-reranker-4b keeps it in its docstring only); one convention.
- pplx serves with `--trust-remote-code`: the GPU wave must assert the remote code imports only what the engine image
  ships (the topk checkpoint's remote config needed `fla`); prefer a plugin-registered config class as topk did.
- jina-embeddings-v5-text-small: replace `anchor: first` with the new last-content declaration (p1-tail 2h).
- qwen3-vl-embedding-2b: served video sampling (fps 2, max_frames 768) vs the reference's (fps 1, max_frames 64) must be
  reconciled in the recipe (one declared policy on both sides, as R20 does for pixels).
- the code-quality sweep's recipe report (research/sweep-recipes/work/report.md, after finalize): every confirmed family-level inconsistency is fixed here, one decision per family (ctxl instruction mode per the paper configs; qwen3-reranker over-cap deviation; NFC-safe render claims; query_max_tokens wording after settle-once; one verified mm_processor_kwargs shape at the v0.31.0 tag for all media recipes; client media fields (image_processor, image_policy) declared so R20's client half holds; requirements-reference.txt everywhere; template file naming, use_activation pin, min_version, client.recipe conventions).

## After the sweep, before RC0 (owner 13:30): deep MiMo adversarial review per core recipe
One MiMo v2.6 Pro (xhigh) reviewer per core recipe (qwen3-embedding-0.6b, qwen3-reranker-0.6b, qwen3-vl-embedding-2b,
qwen3-vl-reranker-2b, zembed-1-embedding, topk-embed-v1-small) and per plugin (topk, pplx): the final recipe against
its model card at the pinned revision, its reference implementation, the vLLM v0.31.0 source at the tag (load path,
pooler, template, media processor), the plugin, and the harness's stage-1 result; evidence-or-not-a-finding as in
research/SWEEP-PROTOCOL.md. The other 12 recipes get the same after the release.

## After the sweep, before RC0 (owner 13:35: one 0.0.1 with everything): deep MiMo adversarial review per recipe
One MiMo v2.6 Pro (xhigh) reviewer per recipe — ALL 18 public recipes — and per plugin (topk, pplx): the final recipe
against its model card at the pinned revision, its reference implementation, the vLLM v0.31.0 source at the tag (load
path, pooler, template, media processor), the plugin, and the harness's stage-1 result; evidence-or-not-a-finding as
in research/SWEEP-PROTOCOL.md. Nothing is deferred past the release.

## Executor and verifiers for this lane (owner decision 14:30)
