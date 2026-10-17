# Third-party licence texts shipped with the `rcp-ndcg-vllm` wheel

`NOTICE` (this distribution's and the repository's) names every third-party work the wheel ships or
ports, with its upstream URL and pinned revision. This directory carries the full text of each licence
that is **not** Apache-2.0, so a redistributor has the terms in hand rather than a pointer.

| file | covers | where the text comes from |
|---|---|---|
| `CC-BY-NC-4.0.txt` | the Jina AI `jina-reranker-v3` reference (`recipes/jina-reranker-v3/reference.py`) | `https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt`, fetched 2026-10-15 |
| `CC-BY-NC-SA-4.0.txt` | the ContextualAI `ctxl-rerank-v2-instruct-multilingual` reference and template (`recipes/ctxl-rerank-v2-instruct-multilingual/{reference.py,template.jinja}`) | `https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt`, fetched 2026-10-15 |
| `MIT.txt` | the Perplexity AI `pplx-embed-v2-context-9b-preview` chunk rule (`models/pplx/chunk_rule.py`) | the SPDX MIT template (`https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt`), with the copyright holder `Perplexity AI` taken from that checkpoint's model-card metadata, because the checkpoint repository ships no LICENSE file (`NOTICE`, "Perplexity pplx-embed-v2-context-9b-preview") |

Apache-2.0 covers everything else; its text is `LICENSE` beside this directory.

The package metadata in `pyproject.toml` declares the combined expression
`Apache-2.0 AND MIT AND CC-BY-NC-4.0 AND CC-BY-NC-SA-4.0`, which is what redistributing this wheel
actually requires: the wheel bundles files under each of those terms. `license-files` lists `LICENSE`,
`NOTICE` and the texts here, so the built wheel carries them.
