"""The shipped judge configs, as package data, loaded by name.

Each ``<name>.yaml`` here is a :class:`~rcp_ndcg.judging.JudgeConfig` for a judge the
paper used or that the documentation walks through: ``gpt_oss_120b``,
``qwen35_397b_nvfp4``, ``qwen35_397b_fp8``, ``qwen36_27b_fp8`` (Qwen3.6-27B), all served by
vLLM on the local host, and ``gpt5_hosted`` (the OpenAI API). They ship inside the
package, so ``JudgeConfig.load("gpt_oss_120b")`` and ``--judge gpt_oss_120b`` work
from any working directory and after ``pip install``. A path to a YAML file works
wherever a name does.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from rcp_ndcg.errors import MissingInputError


def judge_names() -> list[str]:
    """The names of the shipped judge configs, sorted."""
    shipped = files(__name__).iterdir()
    return sorted(entry.name.removesuffix(".yaml") for entry in shipped if entry.name.endswith(".yaml"))


def judge_config_path(value: str | Path) -> Path:
    """The judge config file ``value`` names: a path to a YAML file, else a shipped config's name.

    Raises:
        MissingInputError: ``value`` is neither an existing file nor a shipped name (the hint lists the names).
    """
    path = Path(value)
    if path.suffix in (".yaml", ".yml") or path.exists():
        if not path.is_file():
            raise MissingInputError(f"judge config {value} does not exist")
        return path
    if str(value) in judge_names():
        return Path(str(files(__name__) / f"{value}.yaml"))
    raise MissingInputError(
        f"no judge config named {str(value)!r}",
        hint=f"shipped: {', '.join(judge_names())}; or pass a path to a judge config YAML",
    )


__all__ = ["judge_config_path", "judge_names"]
