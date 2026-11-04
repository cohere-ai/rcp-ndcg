"""The serve argv's flags: every ``--flag`` the 44 variants render is a ``vllm serve`` option at the pinned tag.

``rcp-ndcg-vllm serve --dry-run`` renders the argv without executing it, and a flag the engine dropped takes its
default silently instead of failing, so neither the dry run nor a real serve proves the argv is accepted.  The
gate is a flag-set diff: :data:`VLLM_033_SERVE_FLAGS` is a recorded fact -- the long options the ``vllm serve``
parser registers at ``v0.33.0``, read statically from the tag's own parser sources
(``vllm/entrypoints/common/cli_args.py``'s ``make_arg_parser`` and the ``FrontendArgs`` dataclass it registers
field by field, ``vllm/engine/arg_utils.py``'s ``EngineArgs.add_cli_args`` and
``AsyncEngineArgs.add_cli_args``; ``vllm/entrypoints/cli/serve.py`` registers nothing itself) -- and every
``--flag`` token a rendered argv carries, ``=``-joined or not, must be in it.  The recorded list is the
parser's positive long spellings: vLLM's flexible parser also accepts the underscore spelling of the same
option, which is not a second flag, and the ``--no-<flag>`` negation a ``BooleanOptionalAction`` adds is derived
from its positive spelling (no recipe renders one).  The top-level ``--version`` of
``vllm/entrypoints/cli/main.py`` is not recorded: ``vllm serve`` does not accept it.  An installed plugin may
add options of its own at startup (``load_general_plugins``); this package's entry point registers
architectures, not parser arguments, so no recipe renders one.

The second test is the gate's positive control: a synthetic argv carrying ``--definitely-not-a-flag`` must be
rejected by the same check, and an ``argparse`` parser built from the recorded list must exit non-zero on it.
vLLM is not a dependency of this package (it runs inside the engine image), so the control cannot execute the
engine's own parser here: the parser is built from the record, so the control proves only that the check and a
real ``argparse`` parser reject an unknown token.  The record's own fidelity is the extraction's, verified at
the tag; the real-parser probe runs in the engine image (W13), where ``vllm serve --definitely-not-a-flag``
must exit non-zero.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Sequence

import pytest
from rcp_ndcg_vllm.recipe import Recipe, iter_recipes, serve_argv

_PINNED_VLLM_REF = "v0.33.0"
"""The tag :data:`VLLM_033_SERVE_FLAGS` was read at: the shipped image's tag, ``vllm/vllm-openai:v0.33.0``."""

VLLM_033_SERVE_FLAGS = frozenset(
    {
        "--additional-config",
        "--aggregate-engine-logging",
        "--all2all-backend",
        "--allow-credentials",
        "--allow-deprecated-quantization",
        "--allowed-headers",
        "--allowed-local-media-path",
        "--allowed-media-domains",
        "--allowed-methods",
        "--allowed-origins",
        "--anthropic-disabled-thinking-effort",
        "--api-key",
        "--api-server-count",
        "--async-scheduling",
        "--attention-backend",
        "--attention-config",
        "--aux-output-config",
        "--block-size",
        "--chat-template",
        "--chat-template-content-format",
        "--code-revision",
        "--cohere-format",
        "--cohere-is-reasoning-model",
        "--collect-detailed-traces",
        "--compilation-config",
        "--config",
        "--config-format",
        "--convert",
        "--cp-kv-cache-interleave-size",
        "--cpu-distributed-timeout-seconds",
        "--cpu-offload-gb",
        "--cpu-offload-params",
        "--cudagraph-capture-sizes",
        "--cudagraph-metrics",
        "--custom-histogram-buckets",
        "--data-parallel-address",
        "--data-parallel-backend",
        "--data-parallel-external-lb",
        "--data-parallel-hybrid-lb",
        "--data-parallel-multi-port-external-lb",
        "--data-parallel-rank",
        "--data-parallel-rpc-port",
        "--data-parallel-size",
        "--data-parallel-size-local",
        "--data-parallel-start-rank",
        "--data-parallel-supervisor-port",
        "--dbo-decode-token-threshold",
        "--dbo-prefill-token-threshold",
        "--dcp-comm-backend",
        "--dcp-kv-cache-interleave-size",
        "--dcp-q-replicate",
        "--decode-context-parallel-size",
        "--default-chat-template-kwargs",
        "--default-mm-loras",
        "--device-ids",
        "--device-memory-utilization",
        "--diffusion-config",
        "--disable-access-log-for-endpoints",
        "--disable-cascade-attn",
        "--disable-chunked-mm-input",
        "--disable-custom-all-reduce",
        "--disable-fastapi-docs",
        "--disable-hybrid-kv-cache-manager",
        "--disable-log-stats",
        "--disable-nccl-for-dp-synchronization",
        "--disable-sliding-window",
        "--disable-uvicorn-access-log",
        "--distributed-executor-backend",
        "--distributed-timeout-seconds",
        "--download-dir",
        "--dp-supervisor-probe-failure-threshold",
        "--dp-supervisor-probe-interval-s",
        "--dp-supervisor-probe-timeout-s",
        "--dp-sync-interval",
        "--dtype",
        "--ec-manager-config",
        "--ec-transfer-config",
        "--elastic-ep-max-dp-size",
        "--enable-auto-tool-choice",
        "--enable-batch-sharded-sampling",
        "--enable-chunked-prefill",
        "--enable-cumem-allocator",
        "--enable-dbo",
        "--enable-elastic-ep",
        "--enable-ep-weight-filter",
        "--enable-eplb",
        "--enable-expert-parallel",
        "--enable-fault-tolerance",
        "--enable-flash-late-interaction",
        "--enable-flashinfer-autotune",
        "--enable-force-include-usage",
        "--enable-hc-sp",
        "--enable-layerwise-nvtx-tracing",
        "--enable-log-deltas",
        "--enable-log-outputs",
        "--enable-log-requests",
        "--enable-logging-iteration-details",
        "--enable-lora",
        "--enable-mamba-cache-stochastic-rounding",
        "--enable-mamba-shared-prefix-checkpoint",
        "--enable-mfu-metrics",
        "--enable-mixed-moe-lora-format",
        "--enable-mm-embeds",
        "--enable-moe-shared-loras",
        "--enable-nccl-comm-suspend",
        "--enable-offline-docs",
        "--enable-per-request-metrics",
        "--enable-prefix-caching",
        "--enable-prompt-embeds",
        "--enable-prompt-tokens-details",
        "--enable-request-id-headers",
        "--enable-return-routed-experts",
        "--enable-scale-out",
        "--enable-server-load-tracking",
        "--enable-shm-tensor-arena",
        "--enable-sleep-mode",
        "--enable-ssl-refresh",
        "--enable-tokenizer-info-endpoint",
        "--enable-tower-connector-lora",
        "--enable-trace-replay",
        "--enforce-eager",
        "--engram-config",
        "--eplb-config",
        "--exclude-tools-when-tool-choice-none",
        "--expert-placement-strategy",
        "--fail-on-environ-validation",
        "--fault-tolerance-config",
        "--fingerprint-mode",
        "--fingerprint-value",
        "--fully-sharded-loras",
        "--gdn-prefill-backend",
        "--generation-config",
        "--gpu-memory-utilization",
        "--grpc",
        "--grpc-port",
        "--h11-max-header-count",
        "--h11-max-incomplete-event-size",
        "--headless",
        "--hf-config-path",
        "--hf-overrides",
        "--hf-token",
        "--host",
        "--ignore-patterns",
        "--interleave-mm-strings",
        "--io-processor-plugin",
        "--ir-op-priority",
        "--jit-monitor-mode",
        "--jit-monitor-verbose",
        "--kda-decode-backend",
        "--kda-prefill-backend",
        "--kernel-config",
        "--kv-cache-dtype",
        "--kv-cache-dtype-skip-layers",
        "--kv-cache-memory-bytes",
        "--kv-cache-metrics",
        "--kv-cache-metrics-sample",
        "--kv-events-config",
        "--kv-offloading-backend",
        "--kv-offloading-size",
        "--kv-sharing-fast-prefill",
        "--kv-transfer-config",
        "--language-model-only",
        "--limit-mm-per-prompt",
        "--linear-backend",
        "--load-format",
        "--log-config-file",
        "--log-error-stack",
        "--log-level",
        "--logging-config",
        "--logits-processors",
        "--logprobs-mode",
        "--long-prefill-token-threshold",
        "--long-prefill-token-threshold-adaptive",
        "--lora-dtype",
        "--lora-modules",
        "--lora-target-modules",
        "--mamba-backend",
        "--mamba-block-size",
        "--mamba-cache-dtype",
        "--mamba-cache-mode",
        "--mamba-cache-philox-rounds",
        "--mamba-ssm-cache-dtype",
        "--mamba-ssu-algorithm",
        "--master-addr",
        "--master-port",
        "--max-cpu-loras",
        "--max-cudagraph-capture-size",
        "--max-log-len",
        "--max-logprobs",
        "--max-lora-cls-labels",
        "--max-lora-rank",
        "--max-loras",
        "--max-model-len",
        "--max-num-active-seqs",
        "--max-num-batched-tokens",
        "--max-num-queued-reqs",
        "--max-num-queued-tokens",
        "--max-num-scheduled-tokens",
        "--max-num-seqs",
        "--max-parallel-loading-workers",
        "--media-io-kwargs",
        "--middleware",
        "--mm-device-do-normalize",
        "--mm-encoder-attn-backend",
        "--mm-encoder-attn-dtype",
        "--mm-encoder-fp8-scale-path",
        "--mm-encoder-fp8-scale-save-margin",
        "--mm-encoder-fp8-scale-save-path",
        "--mm-encoder-only",
        "--mm-encoder-tp-mode",
        "--mm-hasher-algorithm",
        "--mm-ipc-gpu-memory-gb",
        "--mm-processor-cache-gb",
        "--mm-processor-cache-type",
        "--mm-processor-device",
        "--mm-processor-kwargs",
        "--mm-shm-cache-max-object-size-mb",
        "--mm-tensor-ipc",
        "--model",
        "--model-class-overrides",
        "--model-impl",
        "--model-loader-extra-config",
        "--moe-backend",
        "--monokernel",
        "--nnodes",
        "--node-rank",
        "--num-gpu-blocks-override",
        "--numa-bind",
        "--numa-bind-cpus",
        "--numa-bind-nodes",
        "--offload-backend",
        "--offload-group-size",
        "--offload-num-in-group",
        "--offload-params",
        "--offload-prefetch-step",
        "--optimization-level",
        "--otlp-traces-endpoint",
        "--override-generation-config",
        "--per-request-spec-decode-metrics",
        "--performance-mode",
        "--pipeline-parallel-size",
        "--pooler-config",
        "--port",
        "--prefill-context-parallel-size",
        "--prefill-schedule-interval",
        "--prefix-cache-retention-interval",
        "--prefix-caching-hash-algo",
        "--prefix-match-unit",
        "--profiler-config",
        "--pt-load-map-location",
        "--quantization",
        "--quantization-config",
        "--ray-workers-use-nsight",
        "--reasoning-config",
        "--reasoning-effort-map",
        "--reasoning-parser",
        "--reasoning-parser-plugin",
        "--renderer-num-workers",
        "--replayssm-buffer-len",
        "--response-role",
        "--return-sampling-mask",
        "--return-tokens-as-token-ids",
        "--revision",
        "--root-path",
        "--runner",
        "--safetensors-load-strategy",
        "--safetensors-prefetch-block-size",
        "--safetensors-prefetch-num-threads",
        "--scheduler-cls",
        "--scheduler-reserve-full-isl",
        "--scheduling-policy",
        "--seed",
        "--served-model-name",
        "--show-hidden-metrics-for-version",
        "--shutdown-timeout",
        "--skip-mm-profiling",
        "--skip-tokenizer-init",
        "--sleep-mode-backend",
        "--sleep-mode-offload-cudagraph",
        "--sleep-preserve-parameter-names",
        "--sparse-indexer-topk-backend",
        "--spec-method",
        "--spec-model",
        "--spec-tokens",
        "--specialize-active-lora",
        "--speculative-config",
        "--sse-keep-alive-interval",
        "--ssl-ca-certs",
        "--ssl-cert-reqs",
        "--ssl-certfile",
        "--ssl-ciphers",
        "--ssl-keyfile",
        "--stream-interval",
        "--structured-outputs-config",
        "--swa-bounded-replay",
        "--tensor-parallel-size",
        "--tokenizer",
        "--tokenizer-mode",
        "--tokenizer-revision",
        "--tokens-only",
        "--tool-call-parser",
        "--tool-parser-plugin",
        "--tool-server",
        "--tool-strict-level",
        "--trust-remote-code",
        "--trust-request-chat-template",
        "--trust-request-mm-kwargs",
        "--ubatch-size",
        "--uds",
        "--use-fp64-gumbel",
        "--use-replayssm",
        "--use-tqdm-on-load",
        "--uvicorn-log-level",
        "--video-pruning-method",
        "--video-pruning-rate",
        "--watchdog-config",
        "--watermark",
        "--watermark-config",
        "--weight-transfer-config",
        "--worker-cls",
        "--worker-extension-cls",
    }
)
"""Every long option the ``vllm serve`` parser registers at :data:`_PINNED_VLLM_REF`, sorted."""


def _unknown_flags(argv: Sequence[str], known: Collection[str] = VLLM_033_SERVE_FLAGS) -> list[str]:
    """The flags ``argv`` carries that ``known`` -- the tag's serve options -- does not register, sorted.

    A ``--flag=value`` token is compared by its flag.  Output: the offending flags (empty when the argv names
    only options the pinned parser registers).
    """
    return sorted({token.split("=", 1)[0] for token in argv if token.startswith("--")} - set(known))


@pytest.mark.parametrize("recipe", iter_recipes(), ids=lambda recipe: recipe.id)
def test_every_flag_a_variant_renders_is_a_serve_option_at_the_pinned_tag(recipe: Recipe) -> None:
    """The full rendered argv -- the template and pooler flags too, not only the engine's -- is at the tag."""
    argv = serve_argv(recipe, port=8000, served_model_name=recipe.id)
    unknown = _unknown_flags(argv)
    assert unknown == [], (
        f"{recipe.id}: the argv renders {unknown}, which {_PINNED_VLLM_REF}'s `vllm serve` parser does not "
        f"register: fix recipe.py/serve.py, or record the flag as engine-version-gated"
    )


def test_an_unknown_flag_is_rejected_by_the_check_the_gate_uses() -> None:
    """Positive control: the check and a parser built from the record both reject a flag the tag lacks."""
    probe = ["vllm", "serve", "example/Model", "--definitely-not-a-flag", "1"]
    assert _unknown_flags(probe) == ["--definitely-not-a-flag"]
    parser = argparse.ArgumentParser()
    for flag in sorted(VLLM_033_SERVE_FLAGS):
        parser.add_argument(flag, nargs="?", default=None)
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--definitely-not-a-flag"])
    assert exit_info.value.code != 0


def test_the_record_is_the_tag_the_product_pins() -> None:
    """A pin bump must force a re-extraction of the record, not silently keep the old set."""
    from rcp_ndcg_vllm.recipe import _PINNED_VLLM_REF as product_pin

    assert _PINNED_VLLM_REF == product_pin, (
        f"re-extract VLLM_033_SERVE_FLAGS at {product_pin}: the record is still the {_PINNED_VLLM_REF} set"
    )
