#!/usr/bin/env python3
"""The judgecheck job's helper: one thinking mode's request block, and the engine media probe.

Two subcommands, both consuming the product's own code (R30):

``modes --judge <id>``
    The thinking modes the recipe documents, one per line as ``<name>\t<extra_body JSON>`` (the body empty
    when the mode is the recipe's shipped client block, so no ``--set`` is passed):

    * the Gemma 4 judges' recipe documents two (``recipes/gemma-4-12b/family.yaml``: thinking is off by
      default, thinking on is a declared client choice with
      ``chat_template_kwargs: {enable_thinking: true}``) -- modes ``off`` and ``on``.  ``--set
      judge.extra_body`` REPLACES the whole block, so the ``on`` body is the recipe's own ``extra_body``
      with the switch merged in (the sampling travels with it).
    * every other judge recipe documents no thinking-mode toggle: the single mode ``default`` runs the
      recipe's shipped client block unchanged.  A family's second mode is added here only with the recipe
      or catalog fact that documents it (never a guessed flag).

``media --judge <id> [--judge-model <id>] [--set judge.*] --out <file>``
    The engine media check for a judge whose family admits media: the product's own probe
    (:meth:`~rcp_ndcg.inference.clients._base.RoleClient.check_engine_media`) sends one prepared 224x224
    image beside its no-media baseline and compares the engine's two ``usage.prompt_tokens`` reports' DELTA
    with the client's counted media tokens.  The probe's policy is the processor family's own stock budget
    (:func:`probe_policy`), so a stock engine keeps the prepared size and the counted tokens describe what
    it sees.  The JSON report is written to ``--out``; a text-only judge is recorded ``checked: false``,
    never silently passed, and a mismatch is the product's typed verdict, not an exception here.

The node script (``judgecheck.sh``) calls both through the client wrapper; a failure writes its report and
exits non-zero so the job records it and moves to the next judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

__all__ = ["judge_modes", "main", "media_report", "probe_policy"]

#: The Gemma 4 family's documented thinking-on toggle (``recipes/gemma-4-12b/family.yaml``: thinking is
#: off by default; on is a declared client choice, the parser reading the template kwarg).  Every other
#: shipped judge recipe documents no thinking-mode toggle: its one mode is the shipped client block.
_GEMMA4_THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True}}

#: The media check's log line carries the engine's delta and the client's count; the report records them.
_DELTA_LINE = re.compile(r"media delta (\d+) equals the counted (\d+)")


def judge_modes(recipe_id: str) -> list[dict[str, Any]]:
    """The thinking modes ``recipe_id`` documents, each with its ``extra_body`` (empty: pass no ``--set``).

    Inputs: a judge recipe id.  Output: the modes in run order -- the Gemma 4 judges' two documented modes
    (``off``, the shipped default, and ``on``, the recipe's declared client choice merged over its own
    ``extra_body``); every other judge's single ``default`` mode, the shipped client block unchanged.
    """
    from rcp_ndcg.judging.judges import judge_config_data

    data = judge_config_data(recipe_id)
    shipped = dict(data.get("extra_body") or {})
    if data.get("image_processor") == "gemma4":
        return [
            {"name": "off", "extra_body": {}},
            {"name": "on", "extra_body": {**shipped, **_GEMMA4_THINKING_ON}},
        ]
    return [{"name": "default", "extra_body": {}}]


def probe_policy(processor: str | None) -> Any:
    """The media probe's declared image policy for a processor family, or ``None`` for a text-only judge.

    The policy is the family's own stock budget (:data:`~rcp_ndcg.data.resolution.PROCESSORS`): a stock
    engine serving the family keeps the prepared size, so the engine's media delta and the client's counted
    tokens describe the same bytes.  Raises ``KeyError`` for a processor family that declares neither a
    pixel range nor a soft-token budget here.
    """
    if processor is None:
        return None
    from rcp_ndcg.data.resolution import PROCESSORS, ImagePolicy

    geometry = PROCESSORS[processor]
    if geometry.resize == "gemma4":
        return ImagePolicy(max_soft_tokens=geometry.soft_tokens)
    return ImagePolicy(min_px=geometry.min_pixels, max_px=geometry.max_pixels)


class _MediaNumbers(logging.Handler):
    """Capture the media check's INFO line (the engine's delta and the client's count)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.line = ""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "engine media check" in message:
            self.line = message


def media_report(config: Any, *, client: Any = None) -> dict[str, Any]:
    """Run the product's engine media check for ``config`` and report its outcome.

    Inputs: the judge config, and (tests only) a client standing in for
    :class:`~rcp_ndcg.judging.client.JudgeClient`.  Output: the report dict -- ``checked: false`` with the
    reason for a text-only judge; else the outcome (``ok`` when the engine's media delta equals the client's
    counted tokens, ``not_checked`` when the reply reported no usage, ``mismatch`` on the product's typed
    media mismatch, ``refused`` when the engine refused the probe), the effective policy, and the two
    numbers when the product's log line carried them.
    """
    policy = probe_policy(config.image_processor)
    if policy is None:
        return {"checked": False, "reason": "text-only"}
    if client is None:
        from rcp_ndcg.judging.client import JudgeClient

        client = JudgeClient.from_config(config)
    client.image_policy = policy.for_processor(config.image_processor)
    capture = _MediaNumbers()
    logger = logging.getLogger("rcp_ndcg.inference.clients._base")
    previous_level = logger.level
    logger.setLevel(logging.INFO)  # the check's INFO line is the only place its two numbers are reported
    logger.addHandler(capture)
    try:
        asyncio.run(client.check_engine_media())
        outcome, detail = "ok", ""
    except Exception as error:  # noqa: BLE001 - the typed verdicts are the report; nothing else may escape
        from rcp_ndcg.errors import CapabilityError, ProviderError

        if isinstance(error, ProviderError):
            outcome, detail = "mismatch", str(error)
        elif isinstance(error, CapabilityError):
            outcome, detail = "refused", str(error)
        else:
            raise
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
    if outcome == "ok":
        recorded = {doc_id for _, doc_id, _, _ in client.media_census.recorded()}
        if "engine_media_check:not_checked" in recorded:
            outcome = "not_checked"
            detail = "the reply reported no usage, so the engine's media delta could not be taken"
    match = _DELTA_LINE.search(capture.line)
    return {
        "checked": True,
        "ok": outcome == "ok",
        "outcome": outcome,
        "detail": detail,
        "processor": config.image_processor,
        "policy": policy.for_processor(config.image_processor).model_dump(mode="json", exclude_none=True),
        "counted": int(match.group(2)) if match else None,
        "reported": int(match.group(1)) if match else None,
    }


def _config(args: argparse.Namespace) -> Any:
    """The judge config the CLI flags name: the same source flags the judging commands take."""
    from rcp_ndcg.cli.judge import JudgeSource

    source = JudgeSource(
        judge=args.judge, judge_url=args.judge_url, judge_model=args.judge_model, set=list(args.set or [])
    )
    return source.judge_config()


def _write(path: str, document: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """The helper's CLI: ``modes`` or ``media`` (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="judgecheck_probe.py", description="the judgecheck job's thinking-mode table and engine media probe"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    modes = commands.add_parser("modes", help="the recipe's documented thinking modes, one per line")
    modes.add_argument("--judge", required=True, help="a judge recipe id")

    media = commands.add_parser("media", help="the engine media check's report")
    media.add_argument("--judge", required=True, help="a judge recipe id")
    media.add_argument("--judge-url", default=None, help="an OpenAI-compatible base URL (.../v1), ad hoc")
    media.add_argument("--judge-model", default=None, help="the served model name")
    media.add_argument("--set", action="append", default=[], help="override judge.<field>: KEY=VALUE")
    media.add_argument("--out", required=True, help="where the JSON report is written")

    args = parser.parse_args(argv)
    if args.command == "modes":
        for row in judge_modes(args.judge):
            print(f"{row['name']}\t{json.dumps(row['extra_body'])}" if row["extra_body"] else row["name"])
        return 0
    try:
        report = media_report(_config(args))
    except Exception as error:  # noqa: BLE001 - the job records the failure and moves to the next judge
        failure = {"checked": False, "ok": False, "outcome": "failed", "detail": f"{type(error).__name__}: {error}"}
        _write(args.out, failure)
        print(f"judgecheck_probe: the media check failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    _write(args.out, report)
    print(json.dumps(report))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":  # pragma: no cover - the job's entry point
    raise SystemExit(main())
