"""Track and classify players in an NFL broadcast clip.

The detector is intentionally still the small YOLO11 model used by the original
project. Most accuracy work happens after detection: field-context and box-
geometry filtering, jersey-region colour/texture features, and temporal
evidence attached to each persistent tracking ID.

The implementation lives in `nfl_analytics.vision`; this file is the CLI.
"""

from __future__ import annotations

import argparse

from nfl_analytics.vision.config import DEFAULT_OUTPUT, DEFAULT_VIDEO
from nfl_analytics.vision.pipeline import VideoOpenError, run_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_VIDEO, help="input MP4 path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output MP4 path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path")
    parser.add_argument(
        "--device", default="auto", help="Ultralytics device (auto, cpu, mps, 0, ...)"
    )
    parser.add_argument(
        "--hide-non-players",
        action="store_true",
        help="do not draw detections rejected as crowd/non-player",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="optional short diagnostic run; 0 processes the full video",
    )
    parser.add_argument(
        "--movement-metrics",
        default="",
        help="optional JSON path for camera and per-track movement diagnostics",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    try:
        summary, _, _ = run_video(
            args.input,
            args.output,
            model_path=args.model,
            device=args.device,
            max_frames=args.max_frames,
            hide_non_players=args.hide_non_players,
            movement_metrics_path=args.movement_metrics or None,
        )
    except VideoOpenError as error:
        raise SystemExit(str(error)) from error

    print("DONE!")
    print(
        f"Processed {summary.frames} frames in {summary.elapsed_seconds:.1f}s "
        f"({summary.processing_fps:.1f} fps)"
    )
    print(f"Detection labels: {dict(summary.label_counts)}")
    print(
        "Camera stabilization: "
        f"{summary.camera_available_frames}/{summary.frames} frames available; "
        f"{summary.camera_estimated_frames} homographies; "
        f"{summary.camera_reset_frames} reference resets"
    )
    if summary.camera_estimated_frames:
        print(
            "Camera matches: "
            f"{summary.camera_feature_total / summary.camera_estimated_frames:.1f} "
            "features/frame; "
            f"{summary.camera_inlier_total / summary.camera_estimated_frames:.1%} mean inliers"
        )
    if args.movement_metrics:
        print(f"Movement metrics: {args.movement_metrics}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    run(parse_args())
