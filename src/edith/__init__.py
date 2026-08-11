"""Edith - a local-first, zero-API-cost autonomous product development agent platform.

Layering (imports flow downward only):

    cli / diagnostics
        -> agents        (contract, registry)
        -> models        (provider abstraction, Ollama implementation)
        -> schemas       (provider-neutral domain types)
        -> config, observability, errors, system
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
