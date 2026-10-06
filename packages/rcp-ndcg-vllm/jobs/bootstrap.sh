#!/usr/bin/env bash
# Superseded: the node's client and reference environments are set up by the rc-build lane's bootstrap
# (GPU-VALIDATION.md, "Node runtime: isolation and resources"): the product's own isolation
# (rcp_ndcg.runners.script.install_argv) installs the client through uvx, and references never run in the
# harness's process. This script installs nothing and runs nothing.
echo "bootstrap.sh is superseded by the rc-build node bootstrap: see GPU-VALIDATION.md, node runtime" >&2
exit 2
