"""Optional Langfuse tracing helpers."""

import os
from collections.abc import Callable
from typing import Any, TypeVar

from langfuse import get_client, observe

F = TypeVar("F", bound=Callable[..., Any])


def langfuse_configured() -> bool:
    """Return whether the minimum Langfuse credentials are available."""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def trace_observation(name: str, as_type: str = "span") -> Callable[[F], F]:
    """Trace a function when Langfuse is configured without capturing its I/O."""
    def decorate(func: F) -> F:
        if not langfuse_configured():
            return func
        return observe(
            name=name,
            as_type=as_type,
            capture_input=False,
            capture_output=False,
        )(func)

    return decorate


def flush_langfuse() -> None:
    """Flush pending observations when tracing is enabled."""
    if langfuse_configured():
        get_client().flush()
