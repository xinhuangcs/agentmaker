"""agentmaker.core.clock: the framework's unified time source, aware UTC (timezone-aware Coordinated Universal Time).

All persisted timestamps across the framework use aware UTC (never naive local time): once written across processes / time zones, ordering stays stable and unambiguous, unaffected by daylight saving time (DST) transitions or the server changing time zone. This is the convention core/message.py chose long ago for conversation timestamps; this module lifts it into a single source of truth shared by all persistence points (memory / checkpoints / bookkeeping / trace, etc.).

When reading back old values or receiving an external naive datetime, ensure_utc normalizes it, avoiding `TypeError: can't compare offset-naive and offset-aware datetimes` when subtracting or comparing aware and naive values.
"""

from datetime import datetime, timezone
from typing import Optional


def now_utc() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to aware UTC: a naive value (no tzinfo) is assumed to be UTC and tagged as such, an already-aware value is converted to UTC, and None is returned unchanged.

    Used on the read side (fromisoformat restoring a naive string from an old database, or a caller passing a naive
    time) to guarantee later comparison / subtraction never mixes types.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
