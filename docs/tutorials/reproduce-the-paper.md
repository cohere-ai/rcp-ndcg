# Reproduce the paper

The scripts in the repository's top-level `experiments/` folder recompute the paper's main tables from the public
datasets. They call no LLM and need no GPU and no credentials. The folder is not part of the installed package, so
run it from a checkout of the repository:

```bash
pip install -r experiments/requirements.txt
python experiments/fetch_data.py                 # the public datasets, at pinned revisions (about 150 MB)
python experiments/run_all.py                    # every check; exits 1 if a value falls outside its tolerance
```

| Script | What it checks against the paper |
|---|---|
| `experiments/leaderboards.py` | the NanoBEIR, BRIGHT and ViDoRe v3 leaderboards (qrel-nDCG@10 and RCP-nDCG@10 of 14 rerankers, every cell) and the TREC-DL reranker pairs |
| `experiments/human_study.py` | the human contest study: counts, the verdict table with its intervals, margin bands and margin AUCs |
| `experiments/external_judges.py` | the external LLM judges on the tournament-order test and on Study 1 |

Each script prints the paper's value next to the reproduced one. Each also runs on its own, for example
`python experiments/leaderboards.py --suite bright`. `experiments/README.md` lists every compared value, the two
documented deviations, and what the scripts do not cover.

## What "reproduce" means here

The scripts score the released rankings of the 14 rerankers with the released gains, under the paper's scoring
protocols ([scoring protocols](../concepts/protocols.md)). The gains come from the released `bt_score`, `theta` and
`gain` columns, which are the outputs of the paper's judging and calibration. The scripts do not refit the
calibration from the judgements, and they do not judge again.

A fresh refit is a different computation. It re-estimates the Bradley-Terry scores and the 2PL parameters from the
released judgements, and a re-judged pool depends on the judge's sampling, which is not bit-deterministic on GPUs.
Neither is expected to reproduce the paper's numbers exactly. To score your own system against the paper's gains,
use the released data as it is ([data](../data.md)).

## Re-judging a pool

Running both judging stages and the calibration yourself needs an OpenAI-compatible endpoint that serves the judge.
The paper used Qwen3.5-397B for NanoBEIR, BRIGHT and ViDoRe v3, and Qwen3.6-27B for TREC-DL. The judge configs
ship in the package (`--judge qwen35_397b_nvfp4`), the paper's engine commands are in
`experiments/paper/serve/` of the repository ([serving](../concepts/serving.md)), and
[calibrate your benchmark](calibrate-your-benchmark.md) walks through a run.
