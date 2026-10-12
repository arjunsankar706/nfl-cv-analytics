"""Smoke tests for the Streamlit dashboard (headless AppTest)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def test_dashboard_renders_without_input():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Pre-Snap Read", "Play-Call Prediction", "Tracking", "Play-by-Play", "Model Lab"]


def test_analyze_without_upload_shows_error():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("Choose a video" in error.value for error in app.error)


def test_invalid_clock_is_rejected():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.text_input[0].input("99:99").run()
    assert any("Clock must be" in error.value for error in app.error)


@pytest.mark.slow
def test_sample_clip_end_to_end(requires_sample_video):
    app = AppTest.from_file(APP, default_timeout=600).run()
    app.radio(key="source").set_value("Sample clip").run()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.session_state["result"] is not None
    assert any("Shotgun" in str(md.value) for md in app.markdown)
