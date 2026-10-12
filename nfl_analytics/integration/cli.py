"""Analyse a broadcast clip end to end and print a JSON report.

    python -m nfl_analytics.integration.cli --input videos/RaidersBroncos.mp4 \\
        --down 2 --distance 22 --quarter 2 --clock 10:48 --score-diff -7

Situation arguments are optional; models that need missing inputs are not
used, and the report says which inputs were missing.
"""

from __future__ import annotations

import argparse
import json
import sys

from nfl_analytics.ml.features import FEATURE_DESCRIPTIONS
from nfl_analytics.vision.pipeline import VideoOpenError

from .video_analysis import analyze_video


def parse_clock(text: str | None) -> float | None:
    if not text:
        return None
    minutes, _, seconds = text.partition(":")
    return float(minutes) * 60 + float(seconds or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-video", default=None, help="optional annotated tracking video")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offense", choices=["A", "B"], default=None, help="A = lighter jerseys, B = darker")
    parser.add_argument("--down", type=int)
    parser.add_argument("--distance", type=float)
    parser.add_argument("--yardline-100", type=float, help="yards from the opponent's end zone")
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--clock", help="game clock in the quarter, MM:SS")
    parser.add_argument("--score-diff", type=float, help="offense score minus defense score")
    parser.add_argument("--off-timeouts", type=int)
    parser.add_argument("--def-timeouts", type=int)
    parser.add_argument("--no-huddle", type=int, choices=[0, 1])
    args = parser.parse_args(argv)

    situation = {
        "down": args.down,
        "ydstogo": args.distance,
        "yardline_100": args.yardline_100,
        "qtr": args.quarter,
        "clock_seconds": parse_clock(args.clock),
        "score_differential": args.score_diff,
        "posteam_timeouts_remaining": args.off_timeouts,
        "defteam_timeouts_remaining": args.def_timeouts,
        "no_huddle": args.no_huddle,
    }
    try:
        result = analyze_video(
            args.input,
            args.output_video,
            model_path=args.model,
            device=args.device,
            max_frames=args.max_frames,
            situation=situation,
            offense_hint=args.offense,
            progress=lambda stage, fraction: print(f"\r{stage:>20s} {fraction:5.0%}", end="", file=sys.stderr),
        )
    except VideoOpenError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    fps = result.tracking.fps
    report = {
        "tracking": {
            "frames": result.tracking.frames,
            "processing_fps": round(result.tracking.processing_fps, 1),
            "device": result.tracking.device,
        },
        "timings": result.timings,
        "messages": result.messages,
        "plays": [],
    }
    for play in result.plays:
        prediction = play.prediction
        report["plays"].append(
            {
                "window": play.window.as_dict(fps),
                "verified_presnap": play.verified,
                "frames_analysed_ok": f"{play.consensus.frames_ok}/{play.consensus.frames_used}",
                "formation": play.consensus.formation_name,
                "offense": play.consensus.offense_team,
                "findings": {name: finding.as_dict() for name, finding in play.consensus.findings.items()},
                "prediction": {
                    "status": prediction.status,
                    "message": prediction.message,
                    "p_pass": None if prediction.p_pass is None else round(prediction.p_pass, 4),
                    "call": prediction.call,
                    "model": prediction.feature_set,
                    "missing_inputs_by_model": {
                        model: [FEATURE_DESCRIPTIONS.get(name, name) for name in missing]
                        for model, missing in prediction.missing_by_model.items()
                    },
                },
            }
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
