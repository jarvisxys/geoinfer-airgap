from __future__ import annotations

from pathlib import Path

import pytest

from geoinfer.service.api.settings import load_settings

CFG = Path(__file__).resolve().parents[2] / "configs" / "service.yaml"


def test_scheduling_env_overrides_and_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEOINFER_DB_URL", "postgresql+psycopg://x@y/z")
    monkeypatch.setenv("GEOINFER_BATCHING", "per_job")
    monkeypatch.setenv("GEOINFER_MAX_WAIT_MS", "50")
    monkeypatch.setenv("GEOINFER_QUEUE", "")  # Compose passes unset variables as ""
    s = load_settings(CFG)
    sc = s.worker.scheduling
    assert (sc.batching, sc.max_wait_ms) == ("per_job", 50.0)
    assert sc.queue == "priority"  # from the config file


def test_settings_require_the_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEOINFER_DB_URL", raising=False)
    with pytest.raises(RuntimeError, match="GEOINFER_DB_URL"):
        load_settings(CFG)
