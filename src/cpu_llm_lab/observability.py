import json
import logging
import os
import time
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Iterator

try:
    from opentelemetry import trace as otel_trace
except ImportError:  # OpenTelemetry is optional in v0.3.
    otel_trace = None


_TRACE_ID: ContextVar[str | None] = ContextVar("cpu_llm_lab_trace_id", default=None)
LOGGER = logging.getLogger("cpu_llm_lab")

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

LOGGER.setLevel(os.getenv("CPU_LLM_LOG_LEVEL", "INFO").upper())


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


def _json_log(event: str, **fields: Any) -> None:
    payload = {"event": event, "trace_id": current_trace_id(), **fields}
    LOGGER.info(json.dumps(payload, ensure_ascii=False, default=str))


@contextmanager
def traced(name: str, **attributes: Any) -> Iterator[str]:
    existing_trace_id = current_trace_id()
    trace_id = existing_trace_id or uuid.uuid4().hex[:16]
    token = _TRACE_ID.set(trace_id) if existing_trace_id is None else None
    start = time.perf_counter()

    if otel_trace is not None:
        tracer = otel_trace.get_tracer("cpu_llm_lab")
        otel_context = tracer.start_as_current_span(name)
    else:
        otel_context = nullcontext()

    with otel_context as span:
        if span is not None and hasattr(span, "set_attribute"):
            for key, value in attributes.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(f"cpu_llm_lab.{key}", value)

        _json_log("span.start", span=name, **attributes)
        try:
            yield trace_id
        except Exception as exc:
            if span is not None and hasattr(span, "record_exception"):
                span.record_exception(exc)
            _json_log(
                "span.error",
                span=name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            _json_log("span.end", span=name, duration_ms=duration_ms, **attributes)
            if token is not None:
                _TRACE_ID.reset(token)
