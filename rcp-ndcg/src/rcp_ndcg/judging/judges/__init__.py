"""The shipped judge configs, as package data, loaded by name; and the recipe route.

The shipped configs here are **vendor profiles**: ``gpt5_hosted`` (the OpenAI API), a
:class:`~rcp_ndcg.judging.JudgeConfig` whose endpoint is the vendor's own. The self-hosted judges
are **serving recipes** in ``rcp-ndcg-vllm``'s package data (``rcp_ndcg_vllm.recipe``), one family
per model family, and resolve through the same path as ``reranker: recipe:<id>``: the recipe's
``client`` block is the judge config (validated by :class:`~rcp_ndcg.judging.JudgeConfig`, R30) and
its ``serve`` block is the engine argv. The three routes a judge may be named by
(``docs/concepts/judges.md``):

* ``--judge <recipe-id>`` or ``judge: recipe:<recipe-id>`` -- a shipped judge recipe, from any
  working directory and after ``pip install``;
* ``--judge recipe:./my-family`` -- a family directory of the operator's own, loaded through the
  recipe schema (marked unshipped, ``status: unverified``, identified by its content hash);
* ``--judge ./my-judge.yaml`` -- a plain :class:`~rcp_ndcg.judging.JudgeConfig` file for any
  OpenAI-compatible endpoint, and ``--judge gpt5_hosted`` -- a shipped vendor profile.

Every route yields the same :class:`~rcp_ndcg.judging.JudgeConfig`; the URL of a self-hosted judge
arrives at runtime (``--set judge.base_url=...``, a run's ``serve:`` block, or ``RCP_NDCG_ENGINES``).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from rcp_ndcg.errors import ConfigError, MissingInputError


def judge_names() -> list[str]:
    """The names of the shipped judge configs (the vendor profiles), sorted."""
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
        hint=f"shipped vendor profiles: {', '.join(judge_names())}; or a judge recipe id (recipe:<id>), "
        "or a path to a judge config YAML",
    )


def judge_config_data(value: str | Path) -> dict[str, Any]:
    """The judge config mapping ``value`` names: a recipe, a shipped vendor profile, or a YAML file.

    Inputs: ``value`` -- a judge recipe id (``gpt-oss-120b``), ``recipe:<id>``, ``recipe:<path>``
    (``recipe:./my-family``, the same form ``reranker: recipe:<id>`` uses), a shipped config's name, or a
    path to a judge config YAML.  Outputs: the parsed mapping the config validates from; for a recipe, the
    recipe's ``client`` block with ``model``/``revision`` injected and ``recipe`` set to the recipe's
    identity (its shipped id, or ``unshipped:sha256:<hex>``).  A recipe names no ``base_url``: a served
    judge's URL arrives at runtime.  Raises :class:`~rcp_ndcg.errors.ConfigError` when ``value`` names a
    recipe of another role (a judge is judged, not encoded), or when rcp-ndcg-vllm is absent (the install
    line); :class:`~rcp_ndcg.errors.MissingInputError` when ``value`` names nothing.
    """
    from rcp_ndcg.inference.recipes import expand_role_recipe, recipe_role, recipe_source

    source = recipe_source(str(value))
    if source is not None:
        role = recipe_role(source)
        if role != "judge":
            raise ConfigError(
                f"recipe {source!r} is a {role} recipe, not a judge recipe",
                hint="a judge is role: judge with client.api: chat; the judge recipes are the ids "
                "`rcp-ndcg-vllm serve` lists (see docs/concepts/judges.md)",
            )
        from rcp_ndcg.judging.client import JudgeConfig

        return expand_role_recipe({"recipe": source}, classes={"judge": JudgeConfig})
    from rcp_ndcg.support.config import load_config

    return load_config(judge_config_path(value))


__all__ = ["judge_config_data", "judge_config_path", "judge_names"]
