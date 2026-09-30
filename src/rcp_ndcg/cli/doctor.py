"""``rcp-ndcg doctor``: what this environment can do, before a long command finds out the hard way.

It reports the Python and package versions, which optional extras are installed (without importing them, so
the check itself stays fast), the cache and runs directories, which credential variables are set (names only,
never values), and, given ``--judge-url``, whether an OpenAI-compatible endpoint answers.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
from typing import Literal

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli.command import command
from rcp_ndcg.support.paths import cache_dir, runs_dir

#: extra -> the import that proves it is installed.
_EXTRAS: dict[str, tuple[str, ...]] = {
    "calibrate": ("torch",),
    "hf": ("huggingface_hub",),
    "data": ("datasets", "pypdfium2"),
    "local": ("transformers", "accelerate"),
    "mteb": ("mteb",),
    "s3": ("s3fs",),
    "azure": ("adlfs",),
    "http": ("aiohttp",),
}

#: Credential variables the shipped configs and hosted backends read (the hosted judge, the Cohere and Voyage
#: rerankers and embedders, the Gemini embedder, gated Hub datasets). Only whether they are set is reported.
_CREDENTIALS = ("OPENAI_API_KEY", "CO_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY", "HF_TOKEN")

Status = Literal["ok", "missing", "fail"]


class DoctorRequest(BaseModel):
    judge_url: str | None = Field(
        default=None, description="An OpenAI-compatible base URL (.../v1) to probe with GET /models."
    )
    timeout_s: float = Field(default=5.0, gt=0, description="Seconds to wait for --judge-url.")


class Check(BaseModel):
    """One finding. ``missing`` is informational (an optional extra); ``fail`` needs attention."""

    name: str
    status: Status
    detail: str = ""


class DoctorReport(BaseModel):
    """The environment, check by check; ``ok`` is false when any check failed."""

    ok: bool
    python: str
    platform: str
    versions: dict[str, str | None]
    cache_dir: str
    runs_dir: str
    checks: list[Check]


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _probe(url: str, timeout_s: float) -> Check:
    import httpx

    target = url.rstrip("/") + "/models"
    try:
        response = httpx.get(target, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return Check(name="judge endpoint", status="fail", detail=f"{target}: {type(exc).__name__}: {exc}")
    if response.status_code in (401, 403):
        detail = f"{target}: HTTP {response.status_code} (check the API key)"
        return Check(name="judge endpoint", status="fail", detail=detail)
    if response.status_code >= 400:
        return Check(name="judge endpoint", status="fail", detail=f"{target}: HTTP {response.status_code}")
    try:
        models = [str(row.get("id")) for row in response.json().get("data", [])]
    except (ValueError, AttributeError):
        models = []
    served = ", ".join(models) or "no models listed"
    return Check(name="judge endpoint", status="ok", detail=f"{target}: serving {served}")


def _text(report: DoctorReport) -> str:
    lines = [
        f"{'python':<15}{report.python} ({report.platform})",
        *(f"{dist:<15}{version or 'not installed'}" for dist, version in report.versions.items()),
        f"{'cache dir':<15}{report.cache_dir}",
        f"{'runs dir':<15}{report.runs_dir}",
        "",
    ]
    lines += [f"{check.status:<8} {check.name:<22} {check.detail}".rstrip() for check in report.checks]
    return "\n".join(lines)


@command("doctor", request=DoctorRequest, result=DoctorReport, text=_text)
def doctor(request: DoctorRequest) -> DoctorReport:
    """Check the environment: versions, installed extras, directories, credentials present, endpoint reachable."""
    versions = {dist: _version(dist) for dist in ("rcp-ndcg", "rcp-ndcg-core", "pydantic", "click", "torch")}
    checks = []
    if versions["rcp-ndcg"] is not None and versions["rcp-ndcg-core"] != versions["rcp-ndcg"]:
        checks.append(
            Check(
                name="rcp-ndcg-core",
                status="fail",
                detail=f"rcp-ndcg {versions['rcp-ndcg']} needs rcp-ndcg-core {versions['rcp-ndcg']}, "
                f"found {versions['rcp-ndcg-core']}",
            )
        )
    for extra, modules in _EXTRAS.items():
        absent = [module for module in modules if not _installed(module)]
        checks.append(
            Check(
                name=f"extra [{extra}]",
                status="missing" if absent else "ok",
                detail=f'not installed: {", ".join(absent)} (pip install "rcp-ndcg[{extra}]")' if absent else "",
            )
        )
    for variable in _CREDENTIALS:
        present = bool(os.environ.get(variable))
        checks.append(
            Check(name=f"${variable}", status="ok" if present else "missing", detail="set" if present else "")
        )
    if request.judge_url is not None:
        checks.append(_probe(request.judge_url, request.timeout_s))
    return DoctorReport(
        ok=not any(check.status == "fail" for check in checks),
        python=platform.python_version(),
        platform=platform.platform(),
        versions=versions,
        cache_dir=str(cache_dir()),
        runs_dir=str(runs_dir()),
        checks=checks,
    )


doctor_cmd: click.Command = doctor

__all__ = ["Check", "DoctorReport", "DoctorRequest", "doctor_cmd"]
