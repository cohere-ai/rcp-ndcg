# Reproduce the paper

The scripts in the repository's top-level `experiments/` folder recompute the paper's main tables from the public
datasets. They call no LLM. They need no GPU and no credentials. The installed package does not include this
folder. Run the scripts from a checkout of the repository:

```bash
pip install ./rcp-ndcg-core ./rcp-ndcg     # or: uv sync
pip install -r experiments/requirements.txt
python experiments/fetch_data.py                 # the public datasets, at pinned revisions (about 150 MB)
python experiments/run_all.py                    # every check; exits 1 if a value falls outside its tolerance
```

| Script | What it checks against the paper |
|---|---|
| `experiments/leaderboards.py` | the NanoBEIR, BRIGHT and ViDoRe v3 leaderboards (qrel-nDCG@10 and RCP-nDCG@10 of 14 rerankers, every cell) and the TREC-DL reranker pairs |
| `experiments/human_study.py` | the human contest study: counts, the verdict table with its intervals, margin bands and margin AUCs |
| `experiments/external_judges.py` | the external LLM judges on the tournament-order test and on Study 1, without the cases the paper's version 2 leaves out |

Each script prints the paper's value next to the reproduced value. Each script also runs alone. For example:
`python experiments/leaderboards.py --suite bright`. The data directory uses revision keys. Thus these checks
never read data that you fetched earlier for arXiv v1. `experiments/README.md` lists every compared value. It
also lists the documented deviation and the items that the scripts do not cover.

## What "reproduce" means here

The scripts score the released rankings of the 14 rerankers with the released gains. They use the paper's scoring
protocols ([scoring protocols](../concepts/protocols.md)). These protocols credit tied scores with their group's
mean gain. The gains come from the released `bt_score`, `theta` and `gain` columns. These columns are the outputs
of the paper's judging and calibration. The scripts do not refit the calibration from the judgements. They also
do not judge again.

A fresh refit is a different computation. It estimates the Bradley-Terry scores and the 2PL parameters again from
the released judgements. A re-judged pool depends on the judge's sampling. This sampling is not bit-deterministic
on GPUs. Do not expect either one to reproduce the paper's numbers exactly. To score your own system against the
paper's gains, use the released data as it is ([data](../data.md)).

## Re-judging a pool

To run both judging stages and the calibration yourself, you need an OpenAI-compatible endpoint that serves the
judge. The paper used Qwen3.5-397B for NanoBEIR, BRIGHT and ViDoRe v3. It used Qwen3.6-27B for TREC-DL. The judge
recipes ship in `rcp-ndcg-vllm` (`--judge recipe:qwen3.5-397b-a17b-nvfp4`). The paper's judges ran on SGLang. The
paper's submission code is the record of those engine commands. This release serves them on vLLM v0.33.0
([serving](../concepts/judges.md)). [calibrate your benchmark](calibrate-your-benchmark.md) gives the full
procedure for a run.
