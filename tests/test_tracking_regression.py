"""End-to-end regression: refactored pipeline must match the original main.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_analytics.vision.pipeline import VideoOpenError, run_video

GOLDEN = Path(__file__).parent / "golden" / "tracking_cpu_90.json"


@pytest.mark.slow
def test_matches_original_pipeline(requires_sample_video, tmp_path):
    golden = json.loads(GOLDEN.read_text())
    summary, frames, metrics = run_video(
        requires_sample_video,
        tmp_path / "out.mp4",
        model_path=str(Path(__file__).parents[1] / "yolo11n.pt"),
        device="cpu",
        max_frames=golden["frames"],
        verbose=False,
    )
    assert summary.frames == golden["frames"] == len(frames)
    assert dict(summary.label_counts) == golden["label_counts"]
    assert metrics["camera"] == golden["camera"]
    assert len(metrics["tracks"]) == golden["track_count"]
    assert (tmp_path / "out.mp4").stat().st_size > 0
    # Structured records expose every drawn detection.
    assert sum(len(frame.observations) for frame in frames) == sum(golden["label_counts"].values())


def test_missing_video_raises_clean_error(tmp_path):
    with pytest.raises(VideoOpenError):
        run_video(tmp_path / "nope.mp4", None)


def test_unsupported_file_raises_clean_error(tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("this is not a video")
    with pytest.raises(VideoOpenError):
        run_video(bogus, None)
