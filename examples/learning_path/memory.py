"""Compatibility alias; implementation lives in reference_code.evidence_assistant."""
from importlib import import_module as _import_module
import sys as _sys

_implementation = _import_module("reference_code.evidence_assistant.memory")
if __name__ == "__main__" and hasattr(_implementation, "main"):
    raise SystemExit(_implementation.main())
_sys.modules[__name__] = _implementation
