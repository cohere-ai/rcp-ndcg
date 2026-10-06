#!/usr/bin/env bash
# Superseded: the node's client and reference environments are set up by the rc-build node bootstrap (the
# product's own isolation, rcp_ndcg.runners.script.install_argv, installs the client through uvx, and
# references run as subprocesses in their own environment). This stub installs nothing and runs nothing.
echo "bootstrap.sh is superseded by the rc-build node bootstrap; run python -m rcp_ndcg_vllm.jobs.run_wave there" >&2
exit 2
