"""Regenerate the README figures from synthetic frames with known ground truth.

    python scripts/make_readme_images.py

No broadcast footage is used, so the images are safe to publish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import synthetic_field as sf  # noqa: E402
from nfl_analytics.formation.analysis import analyze_formation  # noqa: E402
from nfl_analytics.formation.render import draw_formation_overlay, tactical_diagram  # noqa: E402

OUT = ROOT / "docs" / "images"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spots in (("bunch_right", sf.bunch_right()),):
        frame, boxes = sf.render(spots)
        report = analyze_formation(frame, boxes)
        overlay = draw_formation_overlay(frame, report)
        overlay = cv2.resize(overlay, (1280, int(overlay.shape[0] * 1280 / overlay.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{name}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        figure = tactical_diagram(report)
        figure.savefig(OUT / f"{name}_diagram.png", facecolor=figure.get_facecolor(), bbox_inches="tight")
        print(name, report.formation_name)


if __name__ == "__main__":
    main()
