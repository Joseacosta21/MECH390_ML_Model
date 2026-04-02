"""
Shared utilities for the physics package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


def get_or_warn(
    d: Dict[str, Any],
    key: str,
    default: T,
    *,
    context: str = '',
) -> T:
    """
    Like dict.get(), but logs a warning when the fallback default is used.

    Args:
        d:       dictionary to look up (typically the design dict)
        key:     key to retrieve
        default: fallback value if key is missing
        context: optional label for the log message (e.g. 'stresses', 'fatigue')

    Returns:
        d[key] if present, else default (with a logged warning)
    """
    if key in d:
        return d[key]
    where = f' [{context}]' if context else ''
    logger.warning(
        "Key '%s' not found in design dict%s — using fallback default %r",
        key, where, default,
    )
    return default
