"""Redacting URLs for anything that leaves the process: a log line, an error message, a record.

The one redactor of the package (the support layer, so every layer above can use it): :func:`safe_url` for
one URL, :func:`redact_urls` for free text that may carry some (an exception's message).
"""

from __future__ import annotations

import re


def safe_url(url: str) -> str:
    """The form of *url* that may reach a log, an error or a record: userinfo, query and fragment stripped.

    A user may embed credentials in a URL (a documented httpx idiom, ``http://user:key@host``), and a query
    string or a fragment can carry a key too; none of them reaches a log line, an exception message or a run
    manifest (the request itself still uses the full URL). The authority and the path stay as written -- a
    bucket name is case-sensitive and an IPv6 host keeps its brackets. The one redactor: the storage cache's
    messages and the inference layer's logs, errors and engine records all call it.

    Args:
        url: A URL or storage URI; a string without ``://`` (a local path) is returned unchanged.

    Returns:
        ``<scheme>://<host[:port]>[/<path>]``.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        return url  # not an authority-bearing URL: nothing to strip
    rest = rest.partition("#")[0].partition("?")[0]
    authority, _, path = rest.partition("/")
    authority = authority[authority.rfind("@") + 1 :]
    return f"{scheme}://{authority}/{path}" if path else f"{scheme}://{authority}"


#: A URL inside free text (an exception's message): a scheme, ``://``, then everything up to whitespace or a
#: quote.
_URL_IN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"<>]+")


def redact_urls(text: str) -> str:
    """*text* with every URL in it passed through :func:`safe_url`: the form of an exception's message (an
    httpx error names the request's full URL) that may reach a log, a record or a traceback.

    Args:
        text: Free text that may carry URLs.

    Returns:
        The text, each URL's userinfo, query and fragment stripped.
    """
    return _URL_IN_TEXT.sub(lambda match: safe_url(match.group(0)), text)


__all__ = ["redact_urls", "safe_url"]
