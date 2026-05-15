"""OpenTelemetry tracing setup.

In dev and tests we install an in-memory span exporter so spans can be inspected
without external infrastructure. Production wiring (an OTLP exporter) lands in
Phase 7 alongside deployment.
"""

from __future__ import annotations

from typing import Final

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_SERVICE_NAME: Final[str] = "earnings-intelligence-agent"

_configured = False
_memory_exporter: InMemorySpanExporter | None = None


def configure_tracing(*, environment: str = "dev") -> trace.Tracer:
    """Install a :class:`TracerProvider` once per process.

    Returns the tracer for the application service. Calling this more than once
    is a no-op so application startup and unit tests can both invoke it safely.
    """
    global _configured, _memory_exporter
    if _configured:
        return trace.get_tracer(_SERVICE_NAME)

    resource = Resource.create(
        {"service.name": _SERVICE_NAME, "deployment.environment": environment}
    )
    provider = TracerProvider(resource=resource)

    if environment in {"dev", "test"}:
        _memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_memory_exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _configured = True
    return trace.get_tracer(_SERVICE_NAME)


def get_memory_exporter() -> InMemorySpanExporter | None:
    """Return the in-memory exporter when running in dev or test, else ``None``.

    Tests use this to assert that a span was emitted without standing up an
    OpenTelemetry collector.
    """
    return _memory_exporter
