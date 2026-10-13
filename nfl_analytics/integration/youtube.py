"""Fetch a short stretch of a YouTube video for analysis.

Uses the `yt-dlp` command-line tool (install with `uv tool install yt-dlp`
or `pipx install yt-dlp`; current releases need Python 3.10+, so it runs
outside this project's environment). No ffmpeg is needed: the video-only
H.264 stream is downloaded and the requested stretch is cut with OpenCV.

Downloading may be restricted by YouTube's terms and by the video's
copyright; the user is responsible for what they fetch.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
MAX_DURATION_S = 20 * 60
# H.264 first because OpenCV decodes it everywhere; video only, so no merging (no ffmpeg).
FORMAT = "bv*[vcodec^=avc1][height<=720]/bv*[ext=mp4][height<=720]/b[ext=mp4][height<=720]"


class YouTubeError(RuntimeError):
    """A problem the user can act on, phrased for display."""


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and parsed.hostname in YOUTUBE_HOSTS


def find_ytdlp() -> Optional[str]:
    found = shutil.which("yt-dlp")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "yt-dlp"
    return str(fallback) if fallback.exists() else None


def parse_start(text: str) -> float:
    """'1:30' -> 90, '45' -> 45, '' -> 0."""
    text = (text or "").strip()
    if not text:
        return 0.0
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    if seconds < 0:
        raise ValueError(text)
    return seconds


def download(url: str, directory: Path | None = None, timeout_s: int = 600) -> Path:
    if not is_youtube_url(url):
        raise YouTubeError("That doesn't look like a YouTube link. It should start with https://www.youtube.com/ or https://youtu.be/")
    tool = find_ytdlp()
    if tool is None:
        raise YouTubeError("YouTube downloads need the yt-dlp tool. Install it with: uv tool install yt-dlp")
    directory = Path(directory or tempfile.mkdtemp(prefix="di_yt_"))
    command = [
        tool, "--no-playlist", "--no-warnings", "--quiet", "--no-progress",
        "--match-filter", f"duration<={MAX_DURATION_S}",
        "-f", FORMAT,
        "-o", str(directory / "video.%(ext)s"),
        "--", url.strip(),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise YouTubeError("The download took too long. Try a shorter video.") from error
    files = sorted(directory.glob("video.*"))
    if completed.returncode != 0 or not files:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[-1] if detail else "no video file was produced"
        if completed.returncode == 0 and not files:
            reason = f"the video may be longer than {MAX_DURATION_S // 60} minutes"
        raise YouTubeError(f"Couldn't get that video ({reason}).")
    return files[0]


def trim(path: Path, start_s: float, seconds: float, output: Path | None = None) -> Path:
    """Copy [start_s, start_s + seconds) of `path` to a new MP4."""
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise YouTubeError("The downloaded video couldn't be opened.")
        fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first = int(start_s * fps)
        if total and first >= total:
            raise YouTubeError(f"The video is only {total / fps:.0f} seconds long, so it can't start at {start_s:.0f} s.")
        video.set(cv2.CAP_PROP_POS_FRAMES, first)
        output = output or path.with_name("clip.mp4")
        writer = None
        try:
            for _ in range(int(seconds * fps)):
                ok, frame = video.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
        if writer is None:
            raise YouTubeError("No frames could be read from the video.")
        return output
    finally:
        video.release()


def fetch_clip(url: str, start: str = "", seconds: float = 60.0) -> Path:
    """Download a YouTube video and return a clip of `seconds` from `start`."""
    try:
        start_s = parse_start(start)
    except ValueError as error:
        raise YouTubeError("Write the start time like 1:30 (minutes:seconds).") from error
    return trim(download(url), start_s, seconds)
