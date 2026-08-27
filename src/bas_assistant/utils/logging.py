"""Structured logging setup for the CLI/GUI entry points."""

from __future__ import annotations

import logging
import sys

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once; idempotent so scripts can call it safely."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)


__all__ = ["setup_logging"]
