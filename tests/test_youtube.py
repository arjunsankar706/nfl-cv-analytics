"""YouTube link handling and clip trimming (no network access)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nfl_analytics.integration import youtube


@pytest.mark.parametrize(
    "url, ok",
    [
        ("https://www.youtube.com/watch?v=abc123", True),
        ("https://youtu.be/abc123", True),
        ("https://m.youtube.com/watch?v=abc123", True),
        ("http://youtube.com/watch?v=abc123", True),
        ("https://www.youtube.com.evil.example/watch?v=abc123", False),
        ("https://vimeo.com/123", False),
        ("file:///etc/passwd", False),
        ("--exec=rm", False),
        ("", False),
    ],
)
def test_only_youtube_links_are_accepted(url, ok):
    assert youtube.is_youtube_url(url) is ok


def test_non_youtube_link_is_rejected_before_downloading(monkeypatch):
    monkeypatch.setattr(youtube.subprocess, "run", lambda *a, **k: pytest.fail("must not run yt-dlp"))
    with pytest.raises(youtube.YouTubeError, match="YouTube link"):
        youtube.download("https://vimeo.com/123")


@pytest.mark.parametrize("text, seconds", [("", 0), ("45", 45), ("1:30", 90), ("0:00", 0), ("1:02:03", 3723)])
def test_parse_start(text, seconds):
    assert youtube.parse_start(text) == seconds


def test_bad_start_time_is_a_friendly_error():
    with pytest.raises(youtube.YouTubeError, match="1:30"):
        youtube.fetch_clip("https://youtu.be/abc", start="soon")


def write_numbered_video(path, frames: int, fps: float = 10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 2, dtype=np.uint8))
    writer.release()


def test_trim_keeps_the_requested_stretch(tmp_path):
    source = tmp_path / "video.mp4"
    write_numbered_video(source, frames=100)
    clip = youtube.trim(source, start_s=3.0, seconds=2.0)
    video = cv2.VideoCapture(str(clip))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = video.read()
    video.release()
    assert count == 20 and ok
    assert abs(float(first.mean()) - 60) < 6  # frame 30 has brightness 60


def test_trim_past_the_end_is_a_friendly_error(tmp_path):
    source = tmp_path / "video.mp4"
    write_numbered_video(source, frames=20)
    with pytest.raises(youtube.YouTubeError, match="only 2 seconds"):
        youtube.trim(source, start_s=5.0, seconds=2.0)
