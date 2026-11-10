"""run_cli: the scheduler's pipes are UTF-8 whatever the host locale, and an undecodable byte is replaced."""

from __future__ import annotations

import sys

from rcp.runners._cli import run_cli


def test_run_cli_decodes_output_as_utf8_and_replaces_invalid_bytes() -> None:
    """``text=True`` decoded the pipe with the locale's encoding, so a tool emitting a non-UTF-8 byte (or UTF-8
    under a non-UTF-8 locale) crashed the run with UnicodeDecodeError instead of a rendered log."""
    assert run_cli([sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"]) == "\ufffd"
    assert run_cli([sys.executable, "-c", "import sys; sys.stdout.buffer.write('é'.encode())"]) == "é"
