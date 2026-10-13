"""Defensive Intelligence dashboard.

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from nfl_analytics.formation.analysis import UNKNOWN  # noqa: E402
from nfl_analytics.formation.render import draw_formation_overlay, tactical_diagram  # noqa: E402
from nfl_analytics.integration.video_analysis import analyze_video, situation_features  # noqa: E402
from nfl_analytics.ml.features import FEATURE_DESCRIPTIONS  # noqa: E402
from nfl_analytics.ml.predict import STATUS_OK, RunPassPredictor, alignment_to_features  # noqa: E402
from nfl_analytics.vision.device import choose_device  # noqa: E402
from nfl_analytics.vision.pipeline import VideoOpenError  # noqa: E402

import simple_view  # noqa: E402

DETECTOR = ROOT / "yolo11n.pt"
SAMPLE_VIDEO = ROOT / "videos" / "RaidersBroncos.mp4"
REPORT = ROOT / "reports" / "ml_metrics.json"
VIDEO_TYPES = ["mp4", "mov", "m4v", "avi", "mkv"]
IMAGE_TYPES = ["png", "jpg", "jpeg"]
AMBER, CYAN, RED, MUTED = "#ffc83d", "#3db4ff", "#ff5a5a", "#7d8a99"

st.set_page_config(page_title="Defensive Intelligence", page_icon="🏈", layout="wide")
st.markdown(
    """
<style>
  .block-container {padding-top: 3.2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: .02em;}
  .di-header {display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
              border-bottom:1px solid #1f2a36; padding-bottom:.6rem; margin-bottom:1rem;}
  .di-title {font-size:1.7rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase;}
  .di-title span {color:#ffc83d;}
  .di-sub {color:#7d8a99; font-size:.9rem;}
  .chip {display:inline-block; padding:.18rem .55rem; margin:.1rem .25rem .1rem 0; border-radius:4px;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.75rem;
         border:1px solid #2a3644; background:#0f151c; color:#c9d3de;}
  .chip.ok {border-color:#2f6b3f; color:#7ee29a;}
  .chip.warn {border-color:#6b5a2f; color:#ffc83d;}
  .chip.bad {border-color:#6b2f2f; color:#ff7b7b;}
  .card {background:#121a23; border:1px solid #1f2a36; border-radius:8px; padding:.9rem 1rem; height:100%;}
  .card .label {color:#7d8a99; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em;}
  .card .value {font-size:1.45rem; font-weight:750; margin-top:.15rem;}
  .card .note {color:#7d8a99; font-size:.78rem; margin-top:.25rem;}
  .formation {font-size:1.6rem; font-weight:800; color:#ffc83d; letter-spacing:.03em;}
  .probbar {display:flex; height:46px; border-radius:6px; overflow:hidden; border:1px solid #1f2a36;
            font-weight:800; font-size:1.05rem;}
  .probbar div {display:flex; align-items:center; padding:0 .8rem; white-space:nowrap;}
  .run {background:#3db4ff; color:#06121c;} .pass {background:#ffc83d; color:#1c1406; justify-content:flex-end;}
  .muted {color:#7d8a99;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def load_predictor() -> RunPassPredictor:
    return RunPassPredictor.load()


@st.cache_resource(show_spinner=False)
def load_formation_analyzer():
    from nfl_analytics.formation.analyzer import FormationAnalyzer

    return FormationAnalyzer(model_path=str(DETECTOR))


@st.cache_data(show_spinner=False)
def load_report() -> dict | None:
    return json.loads(REPORT.read_text()) if REPORT.exists() else None


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def card(label: str, value: str, note: str = "") -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'


predictor = load_predictor()


def set_view(view: str) -> None:
    st.session_state["view"] = view
    st.rerun()


# The simple view is the default; the detailed view is one click away.
if st.session_state.setdefault("view", "simple") == "simple":
    simple_view.render(predictor, load_formation_analyzer, DETECTOR, SAMPLE_VIDEO, lambda: set_view("detailed"))
    st.stop()

device = choose_device("auto")
report_json = load_report()

# -------------------------------------------------------------------- header
status_chips = [
    chip(f"DETECTOR {device.upper()}", "ok" if device != "cpu" else "warn"),
    chip("YOLO11n", "ok" if DETECTOR.exists() else "bad"),
    chip(f"RUN/PASS MODELS {len(predictor.artifacts)}", "ok" if predictor.available else "bad"),
]
if report_json:
    split = report_json["split"]
    status_chips.append(chip(f"TRAIN {min(split['train'])}-{max(split['train'])} · TEST {min(split['test'])}-{max(split['test'])}"))
st.markdown(
    f"""<div class="di-header"><div><div class="di-title">Defensive <span>Intelligence</span></div>
    <div class="di-sub">Pre-snap formation recognition and run/pass tendencies from broadcast footage</div></div>
    <div>{''.join(status_chips)}</div></div>""",
    unsafe_allow_html=True,
)
if not predictor.available:
    st.warning(
        "No trained RUN/PASS model found in `models/`. Run `python -m nfl_analytics.ml.train --download` "
        "to download nflverse data and train; formation analysis still works without it."
    )

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    if st.button("← Simple view", key="to_simple", use_container_width=True):
        set_view("simple")
    st.markdown("### Input")
    source = st.radio("Source", ["Upload video", "Upload pre-snap image", "Sample clip"], horizontal=False,
                      key="source", label_visibility="collapsed")
    upload = None
    if source == "Upload video":
        upload = st.file_uploader("Broadcast video", type=VIDEO_TYPES)
    elif source == "Upload pre-snap image":
        upload = st.file_uploader("Pre-snap screenshot", type=IMAGE_TYPES)
    elif not SAMPLE_VIDEO.exists():
        st.error("Sample clip not found at videos/RaidersBroncos.mp4.")

    st.markdown("### Options")
    offense_choice = st.selectbox("Offense", ["Auto-detect", "Light jerseys (A)", "Dark jerseys (B)"])
    offense_hint = {"Auto-detect": None, "Light jerseys (A)": "A", "Dark jerseys (B)": "B"}[offense_choice]
    max_frames = st.number_input("Max frames (0 = whole clip)", min_value=0, value=0, step=100)
    write_video = st.toggle("Render annotated tracking video", value=True)

    st.markdown("### Game situation")
    st.caption("Optional. Leave blank if unknown — blanks are never filled in.")
    c1, c2 = st.columns(2)
    down = c1.number_input("Down", min_value=1, max_value=4, value=None, step=1)
    distance = c2.number_input("To go (yd)", min_value=1, max_value=99, value=None, step=1)
    yardline = st.number_input("Yards to opponent end zone", min_value=1, max_value=99, value=None, step=1)
    c3, c4 = st.columns(2)
    quarter = c3.number_input("Quarter", min_value=1, max_value=5, value=None, step=1)
    clock = c4.text_input("Clock (MM:SS)", value="")
    score_diff = st.number_input("Score diff (offense − defense)", min_value=-60, max_value=60, value=None, step=1)
    c5, c6 = st.columns(2)
    off_to = c5.number_input("Off. timeouts", min_value=0, max_value=3, value=None, step=1)
    def_to = c6.number_input("Def. timeouts", min_value=0, max_value=3, value=None, step=1)
    huddle = st.selectbox("No-huddle", ["Unknown", "No", "Yes"])
    run_clicked = st.button("Analyze", type="primary", use_container_width=True, key="analyze")


def parse_clock(text: str):
    if not text.strip():
        return None
    try:
        minutes, _, seconds = text.strip().partition(":")
        value = float(minutes) * 60 + float(seconds or 0)
        return value if 0 <= value <= 900 else None
    except ValueError:
        return None


situation = {
    "down": down,
    "ydstogo": distance,
    "yardline_100": yardline,
    "qtr": quarter,
    "clock_seconds": parse_clock(clock),
    "score_differential": score_diff,
    "posteam_timeouts_remaining": off_to,
    "defteam_timeouts_remaining": def_to,
    "no_huddle": {"Unknown": None, "No": 0, "Yes": 1}[huddle],
}
if clock.strip() and situation["clock_seconds"] is None:
    st.sidebar.error("Clock must be MM:SS within a 15-minute quarter.")


# ------------------------------------------------------------------- running
def save_upload(uploaded) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="di_"))
    path = directory / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())
    return path


if run_clicked:
    st.session_state.pop("result", None)
    st.session_state.pop("image_result", None)
    if source == "Upload pre-snap image":
        if upload is None:
            st.error("Upload a pre-snap screenshot first.")
        else:
            data = np.frombuffer(upload.getbuffer(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                st.error("That file could not be decoded as an image.")
            else:
                with st.spinner("Detecting players and registering the field…"):
                    report = load_formation_analyzer().analyze(image, offense_hint=offense_hint)
                features, _ = situation_features(situation)
                st.session_state["image_result"] = (image, report, predictor.predict({**report.model_inputs(), **features}))
    else:
        path = None
        if source == "Sample clip" and SAMPLE_VIDEO.exists():
            path = SAMPLE_VIDEO
        elif source == "Upload video" and upload is not None:
            path = save_upload(upload)
        if path is None:
            st.error("Choose a video first.")
        else:
            output = Path(tempfile.mkdtemp(prefix="di_out_")) / "tracking.mp4" if write_video else None
            bar = st.progress(0.0, text="Starting…")
            try:
                result = analyze_video(
                    path,
                    output,
                    model_path=str(DETECTOR),
                    device="auto",
                    max_frames=int(max_frames),
                    situation=situation,
                    offense_hint=offense_hint,
                    predictor=predictor,
                    codec="avc1",
                    progress=lambda stage, fraction: bar.progress(min(1.0, fraction), text=stage.title()),
                )
                bar.empty()
                st.session_state["result"] = (str(path), result)
            except VideoOpenError as error:
                bar.empty()
                st.error(f"Unsupported or unreadable video: {error}")


# ----------------------------------------------------------------- rendering
def finding_rows(findings: dict) -> pd.DataFrame:
    order = ["qb_alignment", "backfield", "receiver_distribution", "trips", "bunch", "offensive_line",
             "offense_players_visible", "box_count", "safety_shell", "defensive_front", "defensive_structure",
             "defense_players_visible"]
    rows = []
    for name in order + [n for n in findings if n not in order]:
        if name not in findings:
            continue
        finding = findings[name]
        rows.append({"Read": name.replace("_", " ").title(), "Value": str(finding.value),
                     "Confidence": float(finding.confidence), "Evidence": finding.reason})
    return pd.DataFrame(rows)


def show_findings(findings: dict) -> None:
    st.dataframe(
        finding_rows(findings),
        hide_index=True,
        use_container_width=True,
        column_config={"Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")},
    )


def show_prediction(prediction) -> None:
    if prediction.status == STATUS_OK:
        p_pass = prediction.p_pass
        run_width, pass_width = max((1 - p_pass) * 100, 8), max(p_pass * 100, 8)
        st.markdown(
            f"""<div class="probbar"><div class="run" style="width:{run_width:.1f}%">RUN {1 - p_pass:.0%}</div>
            <div class="pass" style="width:{pass_width:.1f}%">PASS {p_pass:.0%}</div></div>""",
            unsafe_allow_html=True,
        )
        metrics = prediction.test_metrics
        cols = st.columns(4)
        cols[0].markdown(card("Lean", prediction.call, f"{prediction.confidence:.0%} probability"), unsafe_allow_html=True)
        cols[1].markdown(card("Model", prediction.feature_set.replace("_", " + "), prediction.model_name.replace("_", " ")), unsafe_allow_html=True)
        cols[2].markdown(card("Held-out accuracy", f"{metrics.get('accuracy', 0):.1%}", "2024-25 seasons"), unsafe_allow_html=True)
        cols[3].markdown(card("Held-out ROC AUC", f"{metrics.get('roc_auc', 0):.3f}", f"log loss {metrics.get('log_loss', 0):.3f}"), unsafe_allow_html=True)
        st.caption(
            "Probabilities describe league-wide tendencies for this pre-snap picture (2016-2023 training data). "
            "They are not a read of this specific team's play call."
        )
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Inputs used**")
            st.dataframe(
                pd.DataFrame(
                    [{"Feature": FEATURE_DESCRIPTIONS.get(k, k), "Value": v} for k, v in prediction.inputs.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("**What moved the needle** (Δ P(pass) vs. a typical value)")
            contributions = pd.DataFrame(prediction.contributions)
            if not contributions.empty:
                contributions["feature"] = contributions["feature"].map(lambda n: FEATURE_DESCRIPTIONS.get(n, n.replace("_", " ")))
                st.bar_chart(contributions.set_index("feature")["delta_p_pass"], horizontal=True, color=AMBER)
    else:
        st.markdown(
            f'{chip(prediction.status.replace("_", " "), "warn" if prediction.status != "MODEL_NOT_TRAINED" else "bad")}',
            unsafe_allow_html=True,
        )
        st.info(prediction.message)
        if prediction.missing_by_model:
            rows = [
                {"Model": model.replace("_", " + "), "Still needs": ", ".join(FEATURE_DESCRIPTIONS.get(n, n) for n in missing) or "—"}
                for model, missing in prediction.missing_by_model.items()
            ]
            st.markdown("**Inputs each model still needs**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def show_read(report_or_consensus, overlay_bgr, best_report) -> None:
    findings = report_or_consensus.findings
    formation = report_or_consensus.formation_name
    head = st.columns([1.3, 1, 1, 1])
    head[0].markdown(card("Offensive formation", f'<span class="formation">{formation}</span>',
                          "tags shown only above 0.5 confidence"), unsafe_allow_html=True)
    box = findings.get("box_count")
    shell = findings.get("safety_shell")
    qb = findings.get("qb_alignment")
    head[1].markdown(card("QB alignment", str(qb.value if qb else UNKNOWN), f"confidence {qb.confidence:.2f}" if qb else ""), unsafe_allow_html=True)
    head[2].markdown(card("Box count", str(box.value if box else UNKNOWN), f"confidence {box.confidence:.2f}" if box else ""), unsafe_allow_html=True)
    head[3].markdown(card("Safety shell", str(shell.value if shell else UNKNOWN), "pre-snap alignment, not coverage"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        if overlay_bgr is not None:
            st.image(overlay_bgr, channels="BGR", use_container_width=True,
                     caption="Amber = offense, blue = defense. Yellow line = computed line of scrimmage; red = the box.")
    with right:
        if best_report is not None and best_report.status == "OK":
            figure = tactical_diagram(best_report)
            st.pyplot(figure, use_container_width=True)
    show_findings(findings)
    if best_report is not None:
        registration = best_report.registration
        notes = [] if registration is None else registration.notes
        tier = "NONE" if registration is None else registration.tier
        st.markdown(
            chip(f"CALIBRATION {tier}", "ok" if tier == "HOMOGRAPHY" else ("warn" if tier == "YARD_LINES" else "bad"))
            + "".join(chip(note) for note in notes)
            + "".join(chip(w, "warn") for w in best_report.warnings),
            unsafe_allow_html=True,
        )


def model_lab() -> None:
    if not report_json:
        st.info("Train the models to populate this tab: `python -m nfl_analytics.ml.train --download`.")
        return
    st.markdown("#### Held-out evaluation (test seasons never used for fitting or model selection)")
    rows = []
    for feature_set, summary in report_json["feature_sets"].items():
        for name, result in summary["models"].items():
            metrics = result["test"]
            rows.append({
                "Feature set": feature_set, "Model": name + (" ✔" if name == summary["selected_model"] else ""),
                "Accuracy": metrics["accuracy"], "ROC AUC": metrics["roc_auc"], "Log loss": metrics["log_loss"],
                "Brier": metrics["brier"], "ECE": metrics["ece"], "F1 (pass)": metrics["f1_pass"], "Test plays": metrics["n"],
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    left, right = st.columns([1, 1])
    with left:
        calibration = ROOT / "reports" / "calibration_test.png"
        if calibration.exists():
            st.image(str(calibration), caption="Reliability on 2024-25 plays", use_container_width=True)
    with right:
        st.markdown("#### What-if: pre-snap picture → tendency")
        alignment = st.radio("QB alignment", ["SHOTGUN", "PISTOL", "UNDER_CENTER"], horizontal=True)
        box_count = st.slider("Defenders in the box", 3, 10, 6)
        use_situation = st.toggle("Add game situation from the sidebar", value=False)
        inputs = {**alignment_to_features(alignment), "defenders_in_box": float(box_count)}
        if use_situation:
            features, missing = situation_features(situation)
            inputs.update(features)
            if missing:
                st.caption("Sidebar situation is incomplete: " + ", ".join(missing))
        show_prediction(predictor.predict(inputs))


tabs = st.tabs(["Pre-Snap Read", "Play-Call Prediction", "Tracking", "Play-by-Play", "Model Lab"])

image_result = st.session_state.get("image_result")
video_result = st.session_state.get("result")

if image_result is not None:
    image, report, prediction = image_result
    with tabs[0]:
        if report.status != "OK":
            st.warning(report.message)
            st.image(image, channels="BGR", use_container_width=True)
        else:
            show_read(report, draw_formation_overlay(image, report), report)
    with tabs[1]:
        show_prediction(prediction)
    with tabs[2]:
        st.info("Tracking runs on video input. A single image is analysed as one pre-snap frame.")
    with tabs[3]:
        st.info("Play-by-play needs a video with one or more snaps.")
elif video_result is not None:
    path, result = video_result
    verified = result.verified_plays
    with tabs[0]:
        for message in result.messages:
            st.info(message)
        if not verified:
            st.warning("No verified pre-snap alignment in this clip, so no formation read or prediction was produced.")
        else:
            labels = [f"{play.window.as_dict(result.tracking.fps)['start_seconds']:.1f}s – "
                      f"{play.window.as_dict(result.tracking.fps)['end_seconds']:.1f}s · {play.consensus.formation_name}"
                      for play in verified]
            choice = st.selectbox("Pre-snap window", range(len(verified)), format_func=lambda i: labels[i])
            play = verified[choice]
            st.caption(
                f"Consensus of {play.consensus.frames_ok}/{play.consensus.frames_used} analysed frames "
                f"(frames {', '.join(map(str, play.window.analysis_frames))}); offense = team {play.consensus.offense_team}."
            )
            show_read(play.consensus, play.overlay_bgr, play.consensus.best)
    with tabs[1]:
        if verified:
            show_prediction(verified[choice].prediction)
        else:
            st.info("No verified pre-snap window, so there is nothing to predict.")
    with tabs[2]:
        summary = result.tracking
        cols = st.columns(4)
        cols[0].markdown(card("Frames tracked", f"{summary.frames:,}", f"{summary.width}x{summary.height} @ {summary.fps:.1f} fps"), unsafe_allow_html=True)
        cols[1].markdown(card("Throughput", f"{summary.processing_fps:.1f} fps", f"device {summary.device}"), unsafe_allow_html=True)
        cols[2].markdown(card("Camera lock", f"{summary.camera_available_frames / max(1, summary.frames):.0%}", f"{summary.camera_reset_frames} shot resets"), unsafe_allow_html=True)
        cols[3].markdown(card("Analysis time", f"{result.timings.get('total_s', 0):.0f}s", f"tracking {result.timings.get('tracking_s', 0):.0f}s · formation {result.timings.get('formation_s', 0):.0f}s"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.4, 1])
        with left:
            if result.annotated_video and Path(result.annotated_video).exists():
                st.video(result.annotated_video)
            else:
                st.video(path)
        with right:
            st.markdown("**Detections by label**")
            counts = pd.Series(dict(summary.label_counts)).sort_values(ascending=False)
            st.bar_chart(counts, color=CYAN, horizontal=True)
        motion = pd.DataFrame({"frame": np.arange(1, len(result.motion) + 1), "motion": result.motion})
        motion["seconds"] = (motion["frame"] - 1) / summary.fps
        st.markdown("**Player motion after removing camera pan/zoom** (body-heights per second; set alignments sit low, live play high)")
        st.line_chart(motion.set_index("seconds")["motion"], color=AMBER, height=180)
    with tabs[3]:
        rows = []
        for play in result.plays:
            window = play.window.as_dict(result.tracking.fps)
            findings = play.consensus.findings
            rows.append({
                "Start (s)": window["start_seconds"], "Snap frame": window["snap_frame"],
                "Verified": "yes" if play.verified else "no",
                "Formation": play.consensus.formation_name,
                "Box": str(findings["box_count"].value) if "box_count" in findings else "-",
                "Shell": str(findings["safety_shell"].value) if "safety_shell" in findings else "-",
                "P(pass)": None if play.prediction.p_pass is None else round(play.prediction.p_pass, 3),
                "Prediction status": play.prediction.status,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    with tabs[0]:
        st.markdown(
            """
**How it works**

1. **Track** every player (YOLO11 + ByteTrack), stabilise the camera and measure motion.
2. **Find pre-snap windows** — stretches where most players are set — and locate the snap burst.
3. **Register the field** from yard lines and hash marks (image → yards homography), split teams by jersey colour,
   find the line of scrimmage and assign roles.
4. **Read the formation** (QB alignment, backfield, receiver distribution, trips/bunch) and the defense
   (box count, front, safety shell). Uncertain reads stay `UNKNOWN`.
5. **Predict RUN/PASS** with a model trained on 2016-2023 NFL play-by-play and tested on 2024-25.

Choose a source in the sidebar and press **Analyze**. Use the *Model Lab* tab to explore the model without video.
"""
        )
with tabs[4]:
    model_lab()
