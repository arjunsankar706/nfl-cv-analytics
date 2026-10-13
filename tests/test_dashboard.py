"""Smoke tests for the Streamlit dashboard (headless AppTest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def detailed_app(timeout: int = 60) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.session_state["view"] = "detailed"
    return app.run()


def test_simple_view_is_the_default():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Guess the Play" in str(md.value) for md in app.markdown)
    assert not app.tabs


def test_simple_view_asks_for_input_first():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert app.radio(key="kid_source").value == "Paste a YouTube link"
    app.button(key="kid_go").click().run()
    assert any("Paste a YouTube link first" in warning.value for warning in app.warning)
    app.radio(key="kid_source").set_value("Upload my video").run()
    app.button(key="kid_go").click().run()
    assert not app.exception
    assert any("Add a video first" in warning.value for warning in app.warning)


def test_simple_view_rejects_non_youtube_links():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.text_input(key="kid_link").input("https://vimeo.com/123").run()
    app.button(key="kid_go").click().run()
    assert not app.exception
    assert any("doesn't look like a YouTube link" in warning.value for warning in app.warning)


def test_simple_view_always_gives_a_guess():
    from nfl_analytics.formation.analysis import analyze_formation
    from nfl_analytics.ml.predict import RunPassPredictor

    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    report = analyze_formation(blank, [])
    assert report.status != "OK"
    app = AppTest.from_file(APP, default_timeout=60)
    app.session_state["image_result"] = (blank, report, RunPassPredictor({}).predict({}))
    app.run()
    assert not app.exception
    text = " ".join(str(md.value) for md in app.markdown)
    assert "best guess" in text and "PASS" in text and "Not sure" not in text


def test_switching_between_views():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.button(key="to_detailed").click().run()
    assert not app.exception and len(app.tabs) == 5
    # AppTest keeps stale widget state across a full round trip, so the way
    # back is checked from a fresh detailed session.
    app = detailed_app()
    app.button(key="to_simple").click().run()
    assert not app.exception and not app.tabs
    assert any("Guess the Play" in str(md.value) for md in app.markdown)


def test_dashboard_renders_without_input():
    app = detailed_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Pre-Snap Read", "Play-Call Prediction", "Tracking", "Play-by-Play", "Model Lab"]


def test_analyze_without_upload_shows_error():
    app = detailed_app()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert any("Choose a video" in error.value for error in app.error)


def test_invalid_clock_is_rejected():
    app = detailed_app()
    app.text_input[0].input("99:99").run()
    assert any("Clock must be" in error.value for error in app.error)


@pytest.mark.slow
def test_sample_clip_end_to_end(requires_sample_video):
    app = detailed_app(timeout=600)
    app.radio(key="source").set_value("Sample clip").run()
    app.button(key="analyze").click().run()
    assert not app.exception
    assert app.session_state["result"] is not None
    assert any("Shotgun" in str(md.value) for md in app.markdown)
