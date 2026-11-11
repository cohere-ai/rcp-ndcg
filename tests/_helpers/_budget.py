"""The shared default explicit budget for the inference tests.

A self-hosted role config must declare ``tokenizer`` and ``max_tokens``; the test configs declare the saved
test tokenizer, whose path the ``conftest`` fixture fills in here once per session (a served role config is
built with the explicit budget by default; a hosted profile declares neither).
"""

from __future__ import annotations

#: The saved ``tokenizer.json`` path, filled by the ``conftest`` fixture (empty until then).
DEFAULT_TOKENIZER = ""
