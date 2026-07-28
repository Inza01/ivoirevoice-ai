"""Central logging setup."""

from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure a concise application-wide log format."""

    selected_level = (level or os.getenv("IVOIREVOICE_LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, selected_level, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
