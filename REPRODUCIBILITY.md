# Reproducibility

This page says how to recompute the results of
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739)
and what each path needs. There are three levels, from the cheapest to the most expensive.

## 1. Recompute the paper's tables from the released data (no LLM)

The top-level [`experiments/`](experiments/README.md) folder downloads the public datasets and recomputes the
paper's leaderboards, the human contest study and the comparisons with external LLM judges. It needs no GPU, no LLM
and no credentials.

```bash
pip install ./rcp-ndcg-core ./rcp-ndcg     # or: uv sync
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
cover. It also documents the one known deviation: the BRIGHT qrel-nDCG column of TheoremQA Theorems. The pinned
revisions are the datasets' release 2, which carries the numbers of arXiv v2 (see
[below](#tournament-answers-the-papers-code-could-not-parse)). The data directory is keyed by revision
(`<data-dir>/<repo>/<revision>/`), so data fetched earlier for arXiv v1 is never read for these checks.

## 2. Score your own system like the paper (no LLM)

Each benchmark dataset holds the judged candidate pools, the human qrels and the calibrated RCP gains
([docs/data.md](docs/data.md)). Two routes score a system on them:

- this package: `rcp-ndcg eval score --rankings <file> --suite <suite>`, or `rcp_ndcg.eval.evaluate`, with the
  paper's per-suite scoring protocol ([scoring protocols](docs/concepts/protocols.md));
- stock [mteb](https://github.com/embeddings-benchmark/mteb): `ndcg_float_at_10` through the `rcp_ndcg_tasks.py`
  shipped with each dataset ([MTEB integration](docs/how-to/mteb-integration.md)).

Both routes credit tied scores with their group's mean gain on every suite, as the paper does from arXiv v2 on.
arXiv v1's NanoBEIR, BRIGHT and TREC-DL numbers broke ties by a fixed order (document id; judge-pool order) instead;
on untied scores the conventions agree.

## 3. Re-judge a pool with your own LLM endpoint

Recomputing the gains themselves means running both judging stages and the calibration: the Stage A listwise
tournament, the Stage B criteria C1 to C5, and the 2PL fit. This needs an OpenAI-compatible endpoint serving the
judge model and the benchmark corpora. The paper's primary judges were Qwen3.5-397B (NanoBEIR, BRIGHT and ViDoRe
v3; text only) and Qwen3.6-27B (TREC-DL); gpt-oss-120b was the second judge on NanoBEIR, BRIGHT and TREC-DL.

- The first-stage retrievers and rerankers of the paper's runs are served from the shipped recipes:
  `rcp-ndcg-vllm serve <recipe-id>` builds the engine command for the stock `vllm/vllm-openai:v0.31.0` image,
  and a run config's `recipe: <id>` reads the recipe's client block. The release ships 44 recipes in 24 families:
  42 verified end to end on GPU in their waves, and 2 shipped with status failed (the `topk-embed-v1` variants,
  small and xsmall); see the [`rcp-ndcg-vllm` README](rcp-ndcg-vllm/README.md).
- The paper's judges ship the same way: `--judge recipe:<id>` (or `judge: recipe:<id>` in a run config) reads
  the recipe's client block, and `rcp-ndcg-vllm serve <id>` builds its engine command; the package itself keeps
  only the hosted `gpt5_hosted` vendor profile. The paper's
  judges ran on SGLang: the paper's submission code is the record of those engine commands, which this
  repository does not ship; this release serves the same checkpoints on vLLM v0.31.0
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
matches the paper's blank-line join; the TEXT is mteb's, byte for byte (the scoring is `bm25s` on both sides,
and the shipped English config tokenises as mteb's does: the bm25s `en` stop list and the English Snowball
stemmer), and the dense, late-interaction and rerank paths read the retrieval dataloader's join.

### Text limits

See section 3: this package counts the judge's tokens, the paper's code counted characters.

### Over-cap truncation preserves the anchor

The paper's code cut document text without regard to where the model reads its answer (dropping e.g. the trailing
assistant header of a last-token reranker). This package reserves the template's anchors before cutting and
re-attaches the frame after. A recipe declares `reference.known_deviations: [anchor_drop_over_cap]` for the models
whose paper run used the anchor-dropping cut, and the equivalence gates compare under-cap pairs only. Only an
over-cap document can end at a different place.

### Judging defaults

The judge's documents text policy defaults to `keep`: nothing is cut unless a policy is declared. A declared
`truncate`/`fail` policy caps at 32768 tokens by default (`on_overflow` is one of `keep`, `truncate`, `chunk`,
`fail`; every cut recorded in the census; chunk aggregation `max`; media counted in tokens). Every shipped
preset and paper config that relied on the previous
20000-token default pins it explicitly, so no judgement identity moves.

### Judges

The paper's judges are served recipes ([judges](docs/concepts/judges.md)); their divergences from the paper's
judge runs are declared in the recipes' own notes and listed here, because each one moves a re-judged number.
The values are in the recipe's `client` block at its pinned revision:

- **Output caps.** The paper's TREC-DL runs drove the Qwen3.6-27B judge with the deploy strategy's run-level
  `strategy.max_response_tokens: 12288`, which overrode the client's own `max_tokens: 16384`; the second judge
  (gpt-oss-120b) ran the pooled client with the same 12288. The recipes keep their presets' caps:
  `qwen3.6-27b-fp8` sends `max_output_tokens: 16384` and `gpt-oss-120b` sends `8192`, both against the paper's
  12288. A cap is a CONTENT field of the judgement family, so these runs never pool with a paper run: truncation
  and `incomplete` rates can move.
- **Sampling.** The paper's runs set per-run sampling; from the 2026-10-10 owner decision every judge sends its
  checkpoint's own generation defaults, read from its `generation_config.json` at the pinned revision
  (superseding the paper's run-level sampling). `temperature`, `extra_body` and `max_output_tokens` are CONTENT
  fields, so the change re-keys each family (the owner accepted the re-key). The per-judge values are in the
  recipe notes; the gpt-oss-120b variant follows the HF-default rule for a config with no sampling parameters
  (`do_sample: true`).
- **Qwen3.6 revision.** The paper's checkpoint revision for the TREC-DL judge was never recorded (its preset
  carried `revision: null`), so `qwen3.6-27b-fp8` pins the Hub head at the recipe's check date
  (`e89b16ebf1988b3d6befa7de50abc2d76f26eb09`) for a reproducible conformance replay; a different Hub revision
  would move its scores.
- **SGLang to vLLM.** The paper's judges ran on SGLang; this release serves the same checkpoints on vLLM
  v0.31.0 with the recipes' engine blocks (section 3). A different engine moves scores at the bf16 margin and
  re-keys the behaviour fingerprint; the recipe notes record the parser-name differences
  (e.g. vLLM's `openai_gptoss` against SGLang's `gpt-oss`).

## Tournament answers the paper's code could not parse

The code behind the paper (arXiv v1) mishandled tournament answers it could not parse. This release fixes it, and
arXiv v2 of the paper corrects the numbers. The fix changes printed values but none of the paper's conclusions.

**What happened.** Some Stage A answers were valid JSON with a small defect:
- a stray closing brace after the object;
- or LaTeX inside a string field (`\pi`, `\{`), which JSON rejects as an invalid escape.

The paper's parser then fell back to reading document numbers from the whole response text, reasoning included.
In most cases this reproduced the order in which the documents were shown, not the judge's answer. The
Bradley-Terry fit took that order as a complete ranking at five times the weight of a scored window. Windows that
scored only some of their documents entered at the weight of a full window.

**How often.** With the primary judges:
- NanoBEIR: 562 of 140,184 windows, in 272 of 649 queries;
- BRIGHT: 3,224 of 298,723 windows, in 571 of 1,383 queries, mostly LaTeX escapes in AoPS and TheoremQA;
- ViDoRe v3: 6,051 of 3,122,771 windows, in 4,043 of 14,514 queries;
- TREC-DL: 8 of 87,723 windows.

The gpt-oss-120b releases have 38 (NanoBEIR) and 18 (BRIGHT) such windows, and TREC-DL none.

**How arXiv v2 reads the answers.** Each answer is cut at the judge's last reasoning tag, and the text after it is
read with the decoder of this release, which takes the first complete JSON object of that text and never reads
numbers from free text. It recovers the judge's own ranking and scores in 558 (NanoBEIR), 3,086 (BRIGHT), 6,024
(ViDoRe v3) and 6 (TREC-DL) of the windows above. A window enters the fit only when the judge ranked and scored all
of its documents: a complete ranking and a score for every document. The fit leaves out the few windows the decoder
cannot read and the windows that rank or score only some of their documents (40 on NanoBEIR, 403 on BRIGHT, 2,349 on
ViDoRe v3 and 219 on TREC-DL). Stage B answers follow the same rule: after the sixth criterion the paper's rubric
prompt also asked (C6, never fitted) is dropped, an answer that does not give all five criteria for every document
is left out (72 of 64,860 rubric windows on NanoBEIR, 10 of 138,298 on BRIGHT, 11 of 1,451,399 on ViDoRe v3 and 63
of 25,961 on TREC-DL), so Count-nDCG also moves slightly. arXiv v2 then refits the Bradley-Terry
scores and the 2PL calibration of every suite and judge, and recomputes every number that depends on them. Its
tables credit tied scores with their group's mean gain on every suite, where arXiv v1 broke ties by document id or
by judge-pool order. The decoder of this release (parse version 4) makes the cut at the last reasoning tag itself,
so on the stored paper-era answers it decodes the same JSON object as arXiv v2's rule in every window (checked on
the stored NanoBEIR and TREC-DL answers of both stages, 577,449 windows). For Stage A, `rcp-ndcg judge reparse` of
those answers therefore reads the same observations as arXiv v2. For Stage B it does not: the paper's rubric prompt
also asked a sixth criterion (C6), which the paper never fitted and arXiv v2 dropped before requiring C1..C5 of
every document, while the rubric parser refuses a criterion the prompt did not ask (`schema`). The released
`bt_score`, `theta` and `gain` columns remain the reference for reproducing release 2. Live judging with a
reasoning parser in the engine is not affected.

**Effect of the correction on the paper's tables** (RCP-nDCG@10, release 2 against release 1, both scored with
arXiv v1's tie rule, in nDCG points):

| | NanoBEIR | BRIGHT | ViDoRe v3 |
|---|---|---|---|
| Reranker means | +0.10 to +0.23 | +0.09 to +0.98 | 0.00 to +0.02 |
| Largest cell | 1.07 (HotpotQA) | 5.19 (AoPS) | 0.10 |
| Order of the 14 rerankers by mean | unchanged | unchanged | unchanged |
| Reranker pairs per dataset whose significance changes | 12 of 1,183 | 4 of 1,092 | 1 of 728 |
| Significant comparisons that reverse | none | none | none |

The two significance rows compare the per-query scores of the two data releases (paired t-test, p < 0.05). At
this tie rule qrel-nDCG@10 does not change. On TREC-DL, RCP-nDCG@10 separates 61 of the 91 DL19 reranker pairs (60
in arXiv v1) and 70 of the DL20 pairs (unchanged), and it still agrees with NIST on every pair NIST separates (14 of
14, 37 of 37). In the human study the main measure moves from 72.4% of 185 contests to 72.9% of 181.

**Effect of the tie rule.** arXiv v2 credits tied scores with their group's mean gain for qrel-nDCG as well, as the
presets do. On its own, the switch moves each NanoBEIR and BRIGHT table cell by at most 0.21 and 0.35 points of
RCP-nDCG@10 and 1.59 and 0.69 points of qrel-nDCG@10. On TREC-DL, the cells of Qwen3-RR-0.6B, whose scores tie most
often, move by up to 5.10 and 8.63 points. NIST then separates 25 of the DL19 pairs (14 in arXiv v1) and 41 of the
DL20 pairs (37), and RCP-nDCG@10 agrees with NIST on all of them. With both changes, 326 RCP-nDCG@10 and 194
qrel-nDCG@10 cells of `experiments/paper_values.json` differ from arXiv v1.

**Which judged cases enter.** arXiv v2 keeps every collected list and verdict, and a judged case enters an analysis
only if, under the v2 fit, it keeps the property it was selected for. In the external-judge study this leaves out
21 of 480 per-query disagreements, 1 of 120 controls, 9 of 447 ideal-list cases and 1 of 100 minimal pairs. In
Study 1 it leaves out the cases whose reference-judge set no longer has more total RCP gain than the qrel set, 35 of
2,969 cases in all. In the printed analyses these are 3 of 264 NanoBEIR and 30 of 938 BRIGHT cases selected with
Qwen3.5-397B, and 1 of 585 BRIGHT cases selected with gpt-oss-120b (a case without a majority verdict). Release 2 of
`rcp-ndcg-external-validation` flags every one of them (`excluded_under_fit` in `external_judges_cases` and
`study1_cases`), and `experiments/external_judges.py` leaves them out; on release 1, which has no such column, it
keeps every case.

**What changed in the code:**
- One decoder serves both stages. It takes the answer as the text after the last reasoning end tag (`</think>`),
  so an object drafted in the reasoning is never read (parse version 4), removes one code fence, and reads the first
  complete JSON object, ignoring text after it.
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

**Reproducing the paper** uses the released `bt_score`, `theta` and `gain` values, as `experiments/` does. The
revisions it pins are the datasets' release 2: they carry the corrected data and the numbers of arXiv v2, which
`experiments/paper_values.json` transcribes. Release 1 stays loadable at its own revision (the four gain datasets
list both releases in `provenance/releases.json`, `rcp-ndcg-external-validation` in its dataset card) and carries
the numbers of arXiv v1, up to the documented deviations of that time.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -n 4
```

Conformance and the fake-engine replays live in `rcp-ndcg-test` (unpublished) and run per its README.
