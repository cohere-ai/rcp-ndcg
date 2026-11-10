"""Run events: the vocabulary a run publishes, and the subscribers it publishes to.

A run publishes what it does as data, so a dashboard, a ``wandb`` run or a progress line of your own can follow
it without patching the pipeline. The vocabulary is four phases -- :data:`PHASES` -- one :class:`RunEvent` each:
a step's ``step_started``, the ``step_finished`` or ``step_failed`` that closes it, and a ``usage`` event when the
step reported judge usage. The events of a run come in the order the run produces them, which
:class:`Pipeline` guarantees by emitting synchronously as it goes.

A subscriber is a callable that receives the event, installed as an entry point in the ``rcp.events`` group
(``my-dashboard = "my_package.events:subscribe"``); the group is optional and empty by default. A subscriber
that raises is logged once for the run and ignored: a broken dashboard never fails a run, and never changes what
the manifest records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from rcp.errors import ConfigError
from rcp.support.entrypoints import load_target
from rcp.support.logging import get_logger
from rcp.support.urls import redact_urls

__all__ = ["EVENTS_GROUP", "PHASES", "Emitter", "RunEvent", "RunPhase", "Subscriber"]

logger = get_logger(__name__)

#: The entry-point group a run's subscribers are installed in (optional: the product declares none).
EVENTS_GROUP = "rcp.events"

#: What a run publishes: a step began, closed successfully, closed with an error, or reported judge usage.
RunPhase = Literal["step_started", "step_finished", "step_failed", "usage"]

#: The four phases, in the order one step emits them (:data:`RunPhase` as data).
PHASES: tuple[RunPhase, ...] = ("step_started", "step_finished", "step_failed", "usage")

#: A subscriber: a callable that receives each :class:`RunEvent` of a run, in the order the run produces them.
Subscriber = Callable[["RunEvent"], None]


def _now() -> datetime:
    return datetime.now(UTC)


class RunEvent(BaseModel):
    """One moment of a run, as a subscriber receives it.

    Attributes:
        run_id: The run's id (its directory's name).
        step: The step the event belongs to (``retrieve``, ``fuse``, ``rerank``, ``tournament``, ``rubric``,
            ``calibrate``, ``evaluate``).
        phase: What happened: one of :data:`PHASES`.
        at: When the run emitted it (UTC).
        usage: A judging step's counts, as ``rcp.inference.types.Usage`` serialises them (``requests``,
            ``failed_requests``, ``input_tokens``, ``output_tokens``, ``cached_input_tokens``); ``None`` on every
            phase but ``usage``.
        detail: What the phase carries: a finished step's ``status`` and ``duration_s`` (seconds), a failed
            step's ``error`` (the manifest's redacted description), nothing on the other phases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    step: str
    phase: RunPhase
    at: datetime = Field(default_factory=_now)
    usage: dict[str, int] | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Emitter:
    """The subscribers one run publishes to, and the failures it has already reported.

    One emitter per run (:meth:`installed` reads the group), so a subscriber that raises is logged once for the
    run and then left alone instead of writing a line per event. Delivery is synchronous, in installation
    order: a subscriber sees the event before the run continues, so a slow subscriber slows the run (and a
    raising one does not stop the others, nor the run).

    Args:
        subscribers: The callables to publish to, in the order they receive the events.
    """

    def __init__(self, subscribers: Sequence[Subscriber] = ()) -> None:
        self._subscribers = tuple(subscribers)
        self._reported: set[int] = set()

    @classmethod
    def installed(cls) -> Emitter:
        """The emitter of a run: the subscribers the ``rcp.events`` group installs, in entry-point name order.

        A subscriber whose entry point does not load, or that is not callable, is logged once here (once per
        run) and skipped; the rest still receive every event.
        """
        subscribers = []
        for entry_point in sorted(entry_points(group=EVENTS_GROUP), key=lambda entry: entry.name):
            subscriber = _subscriber_of(entry_point)
            if subscriber is not None:
                subscribers.append(subscriber)
        return cls(subscribers)

    @property
    def subscribers(self) -> tuple[Subscriber, ...]:
        """The subscribers this emitter publishes to, in the order they receive the events."""
        return self._subscribers

    def emit(self, event: RunEvent) -> None:
        """Deliver *event* to every subscriber, synchronously and in installation order.

        A subscriber that raises is logged once for this emitter (once per run) and never stops the other
        subscribers or the run: emission is fire-and-forget on the failure, not on the delivery.
        """
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as exc:  # noqa: BLE001 - a subscriber's failure is the subscriber's
                if id(subscriber) in self._reported:
                    continue
                self._reported.add(id(subscriber))
                logger.warning(
                    "[events] the %s subscriber raised %s on %s of %s: %s (logged once; the run continues)",
                    _name_of(subscriber),
                    type(exc).__name__,
                    event.phase,
                    event.step,
                    redact_urls(str(exc)),
                )


def _subscriber_of(entry_point: EntryPoint) -> Subscriber | None:
    """The callable *entry_point* names, or ``None`` (logged) when it does not load or is not callable."""
    try:
        target = load_target(entry_point, kind="run-event subscriber")
    except ConfigError as exc:
        logger.warning("[events] the %s subscriber could not be loaded: %s", entry_point.name, redact_urls(str(exc)))
        return None
    if not callable(target):
        logger.warning(
            "[events] the %s subscriber is a %s, not a callable: a subscriber is a callable that receives the event",
            entry_point.name,
            type(target).__name__,
        )
        return None
    return cast(Subscriber, target)


def _name_of(subscriber: Subscriber) -> str:
    """What a log line calls *subscriber*: its qualified name, else its type's."""
    return getattr(subscriber, "__qualname__", None) or type(subscriber).__name__
