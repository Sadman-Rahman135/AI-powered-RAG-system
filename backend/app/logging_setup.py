"""Process-wide logging: structlog in front of the standard library.

``configure_logging`` renders every record - structlog's and plain ``logging`` ones from libraries alike - as one JSON
object per line when ``LOG_JSON`` is true, or as readable console lines otherwise. Context bound with
``structlog.contextvars`` (the request id) is merged into every line. Uvicorn's own access log is silenced because
the API writes its own access line (method, route template, status, duration, request id - never bodies, query
strings, filenames or document text).

Exception messages are not safe to log: a database error carries its SQL and bound values (invoice fields, file
names), a parser error may quote document text. The API therefore logs failures with ``exception_summary`` - the
exception classes and the code locations, never a message - and console tracebacks are rendered without local
variables.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import structlog

from app.config import Settings

HANDLER_NAME = "docintel"

_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
]


def configure_logging(settings: Settings) -> None:
    """Route all logging through one structlog-formatted handler on the root logger (safe to call again)."""
    final: list[structlog.typing.Processor] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    if settings.log_json:
        final += [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        # renders exc_info itself; plain tracebacks, because the rich ones print local variables (request data)
        final.append(structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback))

    structlog.configure(
        processors=[*_SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(foreign_pre_chain=_SHARED_PROCESSORS, processors=final))
    handler.set_name(HANDLER_NAME)

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if h.get_name() != HANDLER_NAME] + [handler]
    root.setLevel(settings.log_level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def exception_summary(exc: BaseException, *, max_frames: int = 12) -> dict[str, str]:
    """What can be logged about ``exc``: the class of it and of every exception chained to it, and where it was
    raised (``file:line:function`` of the innermost frames). Never a message - see the module docstring."""
    chain: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__module__}.{type(current).__qualname__}")
        current = current.__cause__ or current.__context__
    frames = traceback.extract_tb(exc.__traceback__)[-max_frames:]
    summary = {
        "error_type": chain[0],
        "frames": " > ".join(f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}" for frame in frames),
    }
    if len(chain) > 1:
        summary["error_chain"] = " <- ".join(chain[1:])
    return summary
