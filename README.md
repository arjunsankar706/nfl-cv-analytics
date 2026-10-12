# NFL Computer Vision Analytics

Tracks people in an NFL broadcast with YOLO11n, rejects likely crowd/non-player
detections using the connected playing-field boundary, perspective, broadcast
scene context, and temporal evidence, then classifies persistent tracks as
Raiders, Broncos, or referees. Referee decisions use repeated vertical texture
across the torso rather than black/white pixel counts alone. Classification is
managed by an UNKNOWN/CANDIDATE/ESTABLISHED state machine: low-quality crops do
not vote, new labels require repeated evidence, and established labels use
hysteresis before they can change.

A separate non-decaying identity archive retains the strongest observations
and repeated low-weight team context. Short-term frame scores may be uncertain,
but weak referee-like texture cannot erase a well-supported track identity.

Field position is contextual rather than an absolute gate. Helmet/equipment,
number-like jersey structure, pants and team-uniform consistency, geometry,
field proximity, and crowd density can keep a sideline or closeup player
eligible even when little or no turf appears inside the person's box.

```bash
python main.py
```

The default input is `videos/RaidersBroncos.mp4` and the default comparison
output is `nfl_movement_analytics_v6.mp4`. V5 remains the classification
baseline and is not overwritten.

## Stage 6: player movement

For established Raiders and Broncos tracks, Stage 6 measures the bottom-center
of the person box as an approximate ground/foot point. Sparse optical-flow
features are selected only from the connected field component, with every
person box masked out. A robust homography maps the current broadcast image
into a shot-local reference frame so that camera pan and zoom do not become
player movement.

Foot points are smoothed in that stabilized reference frame. Short tracking
gaps can reconnect; long gaps, implausible jumps, failed field locks, and shot
changes start a new movement run. Sideways/prone boxes and boxes clipped before
the feet are visible are excluded because their bottom center is not a valid
ground point. Distance is accumulated from frame-local displacement after the
camera homography maps the current foot point back into its last observed
image; this prevents long reference-transform drift from inflating speed.
Trails use only a short reference-frame history and are projected back into the
current broadcast view for drawing. The small dashboard reports shot-local
stabilized pixel displacement and pixel speed for the fastest active tracks.

These values are deliberately **not** labeled as yards or mph. Each camera
segment has its own pixel scale. Real-world distance requires a later field
calibration/homography to known yard-line coordinates.

Use `--movement-metrics diagnostics.json` to save camera-lock quality and
per-track, per-segment pixel movement diagnostics. Use `python main.py --help`
for input, output, device, and diagnostic-run options.

Generated videos, model weights, virtual environments, and caches are ignored
by Git.
