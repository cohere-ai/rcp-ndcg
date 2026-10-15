"""Request fields shared by several command groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATASET_HELP = "Dataset URI (hf://, suite:, beir:, jsonl:, images:, videos:, frames:)."


class DatasetInput(BaseModel):
    """``--dataset URI`` with the subset and revision of a Hub dataset."""

    dataset: str = Field(description=DATASET_HELP)
    subset: str | None = Field(default=None, description="The subset of a hf:// or suite: dataset.")
    revision: str | None = Field(default=None, description="The Hub revision of a hf:// or suite: dataset.")

    def load(self) -> Any:
        """The :class:`~rcp_ndcg.data.Dataset`."""
        from rcp_ndcg.data import load_dataset

        return load_dataset(self.dataset, subset=self.subset, revision=self.revision)


def load_yaml_config(path: str, overrides: list[str]) -> dict[str, Any]:
    """A YAML config (``extends:`` resolved) with ``key=value`` overrides applied.

    A retriever or reranker YAML's ``recipe:`` paths are relative to the file that names them, like every
    other relative path in a config (the run config's rule); the overrides' paths stay relative to the
    working directory.
    """
    from rcp_ndcg.support.config import load_config

    return load_config(path, overrides=overrides, relative=_relative_recipe_paths)


def _relative_recipe_paths(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """A retriever or reranker YAML with its ``recipe:`` paths made relative to ``base`` (the file's directory)."""
    from rcp_ndcg.inference.recipes import rebase_recipe_path

    data = dict(data)
    if isinstance(data.get("recipe"), str):
        data["recipe"] = rebase_recipe_path(data["recipe"], base)
    encoder = data.get("encoder")
    if isinstance(encoder, dict) and isinstance(encoder.get("recipe"), str):
        data["encoder"] = {**encoder, "recipe": rebase_recipe_path(encoder["recipe"], base)}
    return data
