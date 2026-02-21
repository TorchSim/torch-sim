"""Stub file for a guaranteed safe import of duecredit constructs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


class InactiveDueCreditCollector:
    """Just a stub at the Collector which would not do anything."""

    def _donothing(self, *_args: Any, **_kwargs: Any) -> None:
        """Perform no good and no bad."""

    def dcite(
        self, *_args: Any, **_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """If I could cite I would."""

        def nondecorating_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return nondecorating_decorator

    active = False
    activate = add = cite = dump = load = _donothing

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


def _donothing_func(*_args: Any, **_kwargs: Any) -> Any:
    """Perform no good and no bad."""
    return None


def _disable_duecredit(exc: Exception) -> None:
    import logging

    logging.getLogger("duecredit").exception(
        "Failed to import duecredit despite being installed: %s", exc
    )


try:
    from duecredit import BibTeX, Doi, Text, Url, due
except Exception as e:  # noqa: BLE001
    if not isinstance(e, ImportError):
        _disable_duecredit(e)
    due = InactiveDueCreditCollector()
    BibTeX = Doi = Url = Text = _donothing_func


def dcite(
    doi: str, description: str | None = None, *, path: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a duecredit decorator from a DOI and description."""
    kwargs: dict[str, Any] = (
        {"description": description} if description is not None else {}
    )
    if path is not None:
        kwargs["path"] = path
    return due.dcite(Doi(doi), **kwargs)
