# Reproducibility

This page says how to recompute the results of
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739)
and what each path needs. There are three levels, from the cheapest to the most expensive.

## 1. Recompute the paper's tables from the released data (no LLM)

The top-level [`experiments/`](experiments/README.md) folder downloads the public datasets and recomputes the
paper's leaderboards, the human contest study and the comparisons with external LLM judges. It needs no GPU, no LLM
and no credentials.

```bash
pip install ./rcp-ndcg-core .           # or: uv sync
pip install -r experiments/requirements.txt
python experiments/fetch_data.py                 # the public datasets, at pinned revisions (about 150 MB)
python experiments/run_all.py
```

Every script prints the paper's value next to the reproduced one and exits with status 1 if a value falls outside
its stated tolerance. They cover:

- the NanoBEIR, BRIGHT and ViDoRe v3 leaderboards (qrel-nDCG@10 and RCP-nDCG@10 of 14 rerankers, every cell);
- the TREC-DL reranker-pair table;
- the human contest study: its counts, the verdict table with its confidence intervals, the margin bands and the
  margin AUCs;
- the three external LLM judges on the tournament-order test and on Study 1.

The scripts score the released reranker runs with the released gains, that is, the `bt_score`, `theta` and `gain`
columns that the paper's judging and calibration produced. They do not refit the calibration and do not judge
again. [`experiments/README.md`](experiments/README.md) lists every compared value and what the scripts do not
cover. It also documents the two known deviations: 10 NanoBEIR RCP-nDCG cells within 0.07 points, and the BRIGHT
qrel-nDCG column of TheoremQA Theorems.

## 2. Score your own system like the paper (no LLM)

Each benchmark dataset holds the judged candidate pools, the human qrels and the calibrated RCP gains
([docs/data.md](docs/data.md)). Two routes score a system on them:

- this package: `rcp-ndcg eval score --rankings <file> --suite <suite>`, or `rcp_ndcg.eval.evaluate`, with the
  paper's per-suite scoring protocol ([scoring protocols](docs/concepts/protocols.md));
- stock [mteb](https://github.com/embeddings-benchmark/mteb): `ndcg_float_at_10` through the `rcp_ndcg_tasks.py`
  shipped with each dataset ([MTEB integration](docs/how-to/mteb-integration.md)).

mteb credits tied scores with their group's mean gain on every suite. The paper's NanoBEIR, BRIGHT and TREC-DL
numbers break ties by a fixed order instead. On untied scores the two conventions agree.

## 3. Re-judge a pool with your own LLM endpoint

Recomputing the gains themselves means running both judging stages and the calibration: the Stage A listwise
tournament, the Stage B criteria C1 to C5, and the 2PL fit. This needs an OpenAI-compatible endpoint serving the
judge model and the benchmark corpora. The paper's primary judges were Qwen3.5-397B (NanoBEIR, BRIGHT and ViDoRe
v3; text only) and Qwen3.6-27B (TREC-DL); gpt-oss-120b was the second judge on NanoBEIR, BRIGHT and TREC-DL.

- The judge configs ship in the package (`rcp-ndcg/src/rcp_ndcg/judging/judges/`, loaded by name), and the paper's engine
  commands, with their images and weights revisions pinned, are in `experiments/paper/serve/`
  ([judges](docs/concepts/judges.md)).
- [Calibrate your benchmark](docs/how-to/calibrate-your-benchmark.md) walks through a run.
- The schedules are specified in placements per document, so their window counts scale with the pool. At the
  paper's pool of 150 candidates the defaults give exactly its counts: 53 random, 27 stratified (both mirrored) and
  7 x 8 adaptive tournament windows of 10 (216 calls), and 100 rubric windows of 10, 50 of them random. The page-image
  and video tournament keeps the placements in windows of 5: 106 random and 54 stratified windows at 150, and 7 x 16
  adaptive windows (the placements' exact count at that window). No reduced TREC-DL schedule is configured in this
  package; one would be stated in placements the same way.
- The paper's judging limited document text by characters: a window's share of the judge's context was converted
  to characters at 2.0 characters per token. This package counts the judge's own tokens (`judge.tokenizer`) and cuts
  at token boundaries ([preprocessing](docs/concepts/preprocessing.md)). A document that fits under both rules is
  shown in full either way. Only a document long enough to be cut can end at a different place, so the difference
  matters only when re-judging long documents like the paper.
- LLM serving on GPUs is not bit-deterministic. A re-judged pool is therefore not expected to match the released
  gains exactly, and neither is a refit of the calibration from the released judgements.

Cohere Embed v4 (one of the three first-stage retrievers) and the Cohere rerankers are called through the public
Cohere API and read `CO_API_KEY` (or `COHERE_API_KEY`) from the environment.

## Deviations from the paper's code

Every cause of a number that moves between the paper's tables and a reproduction from this code, in one ledger.

### Tournament answers

See [Tournament answers the paper's code could not parse](#tournament-answers-the-papers-code-could-not-parse).

### Documents read MTEB's title join

The paper's code joined a document's title to its body with a **blank line** at read time (`title\n\nbody`), so its
models read the blank line; a document without a title was read as its body, unstripped. This package keeps the
title as its own field (`Document.title`, nothing joins at read time) and applies the join where a model's text is
formatted, with the rule of mteb's retrieval dataloader, byte for byte: `(title + " " + body).strip()`, the body
alone (stripped) without a title. The paper's published runs therefore read a different string for every document
that carries a title; the released tables are unaffected (they score stored runs), but a re-scored or re-judged
pool reads mteb's join, as the [MTEB integration](docs/how-to/mteb-integration.md) requires. A model or recipe that
takes the title separately declares `title: separate` on its role config.

The **sparse (BM25) path** is a second, declared divergence: mteb's own BM25 is not a served model and reads no
dataloader, so the sparse path follows mteb's BM25 instead -- a corpus row is indexed as `title + "\n" + body` (a
newline, both as given) and a query is the per-query instruction's append alone, with no `Task:` frame. Neither
matches the paper's blank-line join; both are mteb's, byte for byte, and the dense, late-interaction and rerank
paths read the retrieval dataloader's join.

### Text limits

See section 3: this package counts the judge's tokens, the paper's code counted characters.

### Over-cap truncation preserves the anchor

The paper's code cut document text without regard to where the model reads its answer (dropping e.g. the trailing
assistant header of a last-token reranker). This package reserves the template's anchors before cutting and
re-attaches the frame after. A recipe declares `reference.known_deviations: [anchor_drop_over_cap]` for the models
whose paper run used the anchor-dropping cut, and the equivalence gates compare under-cap pairs only. Only an
over-cap document can end at a different place.

### Judging defaults

The judge's documents text policy defaults to 32768 tokens (`on_overflow` default `cut`, every cut recorded; chunk
aggregation `max`; media counted in tokens). Every shipped preset and paper config that relied on the previous
20000-token default pins it explicitly, so no judgement identity moves.

## Tournament answers the paper's code could not parse

The code behind the paper (arXiv v1) mishandled tournament answers it could not parse. This release fixes it. The
fix changes printed values but none of the paper's conclusions. A minor correction to the paper is forthcoming.

**What happened.** Some Stage A answers were valid JSON with a small defect:
- a stray closing brace after the object;
- or LaTeX inside a string field (`\pi`, `\{`), which JSON rejects as an invalid escape.

The paper's parser then fell back to reading document numbers from the whole response text, reasoning included.
In most cases this reproduced the order in which the documents were shown, not the judge's answer. The
Bradley-Terry fit took that order as a complete ranking at five times the weight of a scored window. Windows that
scored only some of their documents entered at the weight of a full window.

**How often.** With the Qwen3.5-397B judge:
- NanoBEIR: 562 of 140,184 windows, in 272 of 649 queries;
- BRIGHT: 3,224 of 298,723 windows, in 691 of 1,383 queries, mostly LaTeX escapes in AoPS and TheoremQA.

The gpt-oss-120b releases have 38 (NanoBEIR) and 18 (BRIGHT) such windows, and TREC-DL at most 8.

**Effect on the paper's tables.** Estimated with the judges' own answers recovered from the raw responses, and the
2PL parameters held at their released values:

| | NanoBEIR | BRIGHT |
|---|---|---|
| Reranker means | +0.12 to +0.25 pp | +0.35 to +1.14 pp |
| Largest cell | 1.05 pp | 5.6 pp (AoPS) |
| System ranking | unchanged | unchanged |
| Significant comparisons that reverse | none | none |
| Significance decisions that change (all near p = 0.05) | 13 of 1,183 | 9 of 1,092 |

**What changed in the code:**
- One decoder serves both stages. It removes reasoning blocks and one code fence, and reads the first complete JSON
  object, ignoring text after it.
- It repairs only a stray quote after a number, and keeps an invalid escape inside a string as literal text. Every
  ranking, score and criterion it returns is the judge's own. It never reads numbers from free text and never fills
  in missing documents.
- Judges configured with `decoding: json_schema` receive each stage's answer schema as `response_format`, and the
  judgement family records the decoding.
- An answer that still fails is stored as an invalid window with a category (`truncated`, `no_json`,
  `invalid_json`, `schema`, `incomplete`, `refused`), and its raw text is kept. `rcp-ndcg judge reparse` re-reads
  stored answers with the current parser without calling the judge.
- Calibration reports invalid windows per query. It warns when a query loses more than 5% of a stage or any
  adaptive window, and refuses under `--strict`.
- A window without scores for all its documents is invalid and not fitted.

**Reproducing the paper as published** uses the released `bt_score`, `theta` and `gain` values, as `experiments/`
does. They carry the paper's numbers. A corrected data revision will accompany the paper's correction.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -n 4
```

Conformance and the fake-engine replays live in `rcp-ndcg-test` (unpublished) and run per its README.
