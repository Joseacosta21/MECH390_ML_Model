"""
Shared utilities for the physics package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TypeVar

T = TypeVar('T')

logger = logging.getLogger(__name__)


# looks up key in dict; logs a warning when the fallback default is used
def get_or_warn(
    d: Dict[str, Any],
    key: str,
    default: T,
    *,
    context: str = '',
) -> T:
    if key in d:
        return d[key]
    where = f' [{context}]' if context else ''
    logger.warning(
        "Key '%s' not found in design dict%s - using fallback default %r",
        key, where, default,
    )
    return default
