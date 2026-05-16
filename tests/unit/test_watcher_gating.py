"""Watcher gating behind ``watcher_mode_enabled``."""
from __future__ import annotations

import pytest


def test_ensure_watcher_enabled_when_off() -> None:
    from app.agents.watcher import WatcherDisabledError, ensure_watcher_enabled

    with pytest.raises(WatcherDisabledError):
        ensure_watcher_enabled(enabled=False)


def test_ensure_watcher_enabled_when_on() -> None:
    """No exception when the flag is on."""
    from app.agents.watcher import ensure_watcher_enabled

    ensure_watcher_enabled(enabled=True)
