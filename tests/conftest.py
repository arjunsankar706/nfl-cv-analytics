from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_VIDEO = ROOT / "videos" / "RaidersBroncos.mp4"
DETECTOR = ROOT / "yolo11n.pt"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs YOLO on real footage (needs local video + weights)")


@pytest.fixture
def requires_sample_video():
    if not SAMPLE_VIDEO.exists() or not DETECTOR.exists():
        pytest.skip("sample broadcast video or yolo11n.pt not available locally")
    return SAMPLE_VIDEO
