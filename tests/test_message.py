"""Message type regression (hermetic): runtime role validation (Literal only guards statically), default timestamp is tz-aware UTC, and to_dict shape."""

from datetime import timezone

import pytest

from agentmaker.core import Message


def test_valid_roles_accepted():
    for role in ("user", "assistant", "system", "tool"):
        assert Message("x", role).role == role


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        Message("x", "useer")           # a misspelled role raises at runtime (silently passed before the fix)


def test_default_timestamp_is_utc_aware():
    m = Message("x", "user")
    assert m.timestamp.tzinfo is timezone.utc   # tz-aware UTC, not naive local time


def test_explicit_timestamp_preserved():
    from datetime import datetime
    ts = datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)
    assert Message("x", "user", timestamp=ts).timestamp == ts


def test_to_dict_drops_extras():
    assert Message("hi", "user").to_dict() == {"role": "user", "content": "hi"}
