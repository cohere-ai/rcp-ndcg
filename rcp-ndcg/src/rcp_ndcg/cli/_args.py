"""Request fields shared by several command groups."""

from __future__ import annotations

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
    """A YAML config (``extends:`` resolved) with ``key=value`` overrides applied."""
    from rcp_ndcg.support.config import load_config

    return load_config(path, overrides=overrides)
