"""Strict validation of in-memory records against a row model, with errors that name the record and the key.

:meth:`~rcp_ndcg.data.Dataset.from_records` and :meth:`~rcp_ndcg.data.Rankings.from_records` read plain dicts (or
the row models themselves) through :func:`validate_rows`; an unknown key, a missing field or a value of the wrong
type is a :class:`~rcp_ndcg.errors.DataError` whose ``details`` carry the record's position, the field, the given
value and, for an unknown key, the closest known field.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from rcp_ndcg.errors import DataError


def validate_rows[Row: BaseModel](
    model: type[Row], records: Iterable[Mapping[str, Any] | BaseModel], *, what: str
) -> list[Row]:
    """Every record as a ``model`` instance.

    Args:
        model: The row model (``extra="forbid"``).
        records: Dicts, or instances of ``model``.
        what: What the records are, for the error message (``"queries"``, ``"rankings"``).

    Returns:
        The validated rows, in input order.

    Raises:
        DataError: A record is not a mapping, has an unknown key, lacks a field or holds a wrong value.
    """
    rows: list[Row] = []
    for index, record in enumerate(records):
        if isinstance(record, model):
            rows.append(record)
            continue
        if isinstance(record, BaseModel):
            record = record.model_dump()
        if not isinstance(record, Mapping):
            raise DataError(
                f"{what}[{index}] is a {type(record).__name__}, not a record (a dict or a {model.__name__})",
                details={"records": what, "index": index},
            )
        try:
            rows.append(model.model_validate(dict(record)))
        except ValidationError as exc:
            raise _row_error(exc, model, what, index, record) from None
    return rows


def _row_error(
    exc: ValidationError, model: type[BaseModel], what: str, index: int, record: Mapping[str, Any]
) -> DataError:
    known = sorted(set(model.model_fields) | {f.alias for f in model.model_fields.values() if f.alias})
    problems = []
    for error in exc.errors(include_url=False):
        field = ".".join(str(part) for part in error["loc"]) or "<record>"
        message = error["msg"].removeprefix("Value error, ")
        problem: dict[str, Any] = {"field": field, "problem": message, "input": _jsonable(error.get("input"))}
        if error["type"] == "extra_forbidden":
            problem["problem"] = "unknown key"
            close = difflib.get_close_matches(str(error["loc"][-1]), known, n=1)
            if close:
                problem["did_you_mean"] = close[0]
        elif error["type"] == "missing":
            problem["input"] = None
        problems.append(problem)
    first = problems[0]
    guess = f" (did you mean {first['did_you_mean']!r}?)" if "did_you_mean" in first else ""
    return DataError(
        f"{what}[{index}]: {first['field']}: {first['problem']}{guess}",
        hint=f"a {what} record has the keys {known}",
        details={
            "records": what,
            "index": index,
            **{key: _jsonable(record[key]) for key in _IDENTIFYING if key in record},
            "errors": problems,
        },
    )


_IDENTIFYING = ("system", "dataset", "query_id", "doc_id")
"""Keys that locate a failing record, copied into the error's details."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


__all__ = ["validate_rows"]
