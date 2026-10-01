"""Tests for shared-secret configuration and Streamlit sign-in."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.utils.auth import configured_api_key

TEST_API_KEY = "streamlit-test-key-with-more-than-32-characters"
APP_FILE = Path(__file__).parents[1] / "app" / "main.py"


def test_configured_api_key_requires_a_strong_secret(monkeypatch) -> None:
    """Missing or short shared keys fail closed."""
    monkeypatch.delenv("INSIGHT_API_KEY", raising=False)
    assert configured_api_key() is None
    monkeypatch.setenv("INSIGHT_API_KEY", "too-short")
    assert configured_api_key() is None
    monkeypatch.setenv("INSIGHT_API_KEY", TEST_API_KEY)
    assert configured_api_key() == TEST_API_KEY


def test_streamlit_sign_in_accepts_valid_key(monkeypatch) -> None:
    """The Streamlit gate accepts the configured key and records the session."""
    monkeypatch.setenv("INSIGHT_API_KEY", TEST_API_KEY)

    app_test = AppTest.from_file(str(APP_FILE), default_timeout=20).run(timeout=20)
    app_test.text_input[0].set_value(TEST_API_KEY)
    app_test.button[0].click().run(timeout=20)

    assert app_test.session_state["authenticated"] is True


def test_streamlit_gate_shows_setup_error_when_key_missing(monkeypatch) -> None:
    """The UI does not expose data pages without a configured key."""
    monkeypatch.delenv("INSIGHT_API_KEY", raising=False)

    app_test = AppTest.from_file(str(APP_FILE), default_timeout=20).run(timeout=20)

    assert any("Authentication is not configured" in item.value for item in app_test.error)
