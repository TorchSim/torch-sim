"""Helpers for optional duecredit citations."""

from collections.abc import Callable
from typing import Any


def _noop_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    return func


try:
    from duecredit import Doi, due
except ImportError:

    def dcite(
        doi: str, description: str | None = None, *, path: str | None = None
    ) -> Callable:
        """Return a no-op decorator when duecredit is unavailable."""
        del doi, description, path
        return _noop_decorator
else:

    def dcite(
        doi: str, description: str | None = None, *, path: str | None = None
    ) -> Callable:
        """Create a duecredit decorator from a DOI and description."""
        kwargs: dict[str, Any] = (
            {"description": description} if description is not None else {}
        )
        if path is not None:
            kwargs["path"] = path
        return due.dcite(Doi(doi), **kwargs)
