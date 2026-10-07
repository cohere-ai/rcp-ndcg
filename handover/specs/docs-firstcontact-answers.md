<!-- Handover copy of the operator's working note `research/docs-firstcontact/work/OPERATOR-ANSWERS.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Operator answers to docs-firstcontact's open questions (17:58; binding for the docs lane and the layout move)
- Q1: the mapping form on the existing `recipe:` field — a role config that names `recipe: <id>` takes its whole client
  block (api, tokenizer, budgets, template, media, instruction mode) from the recipe; `base_url` (and other RUNTIME
  fields) stay on the config; any CONTENT field set explicitly must equal the recipe's or is refused (ConfigError naming
  both values). CLI shorthand `--reranker recipe:<id>` / `--retriever recipe:<id>` expands to that mapping, with the
  URL from `--set ...base_url=` or serve-by-role. Docs show the mapping, mention the shorthand.
- Q2: yes — `serve:` may use `command: ["rcp-ndcg-vllm", "serve", "<id>"]`; the `--no-deps` install is image
  preparation in the docs, with the inline `pip install --no-deps ... && exec rcp-ndcg-vllm serve` form shown for stock
  images.
- Q3-Q9: the recommended defaults (Q3-D ... Q9-D) as written in the blueprint.
