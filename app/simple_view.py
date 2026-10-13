"""Simple view of the dashboard: big text, few colours, plain words.

Written so a young football fan can follow it. It shows the same analysis
and the same model probabilities as the detailed view, only explained
differently; when the computer cannot tell, it says so.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import streamlit as st

from nfl_analytics.integration.video_analysis import analyze_video
from nfl_analytics.integration.youtube import YouTubeError, fetch_clip
from nfl_analytics.ml.predict import STATUS_OK, RunPassPredictor
from nfl_analytics.vision.pipeline import VideoOpenError

CLIP_SECONDS = 60

CSS = """
<style>
  .block-container {max-width: 980px; padding-top: 3.2rem;}
  html, body, [class*="st-"] {font-size: 1.12rem;}
  .kid-title {font-size: 2.6rem; font-weight: 800; text-align: center; margin: 0;}
  .kid-sub {font-size: 1.3rem; text-align: center; color: #b7c2cf; margin: .3rem 0 1.6rem;}
  .kid-step {font-size: 1.5rem; font-weight: 750; margin: 1.6rem 0 .5rem;}
  .kid-box {background: #141c26; border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: .8rem;
            font-size: 1.25rem; line-height: 1.6;}
  .kid-box b {color: #ffd76a;}
  .kid-guess {text-align: center; background: #141c26; border-radius: 20px; padding: 1.6rem 1rem;}
  .kid-guess .word {font-size: 4rem; font-weight: 850; letter-spacing: .04em; color: #ffd76a; line-height: 1.1;}
  .kid-guess .why {font-size: 1.35rem; margin-top: .6rem;}
  .kid-dots {font-size: 2.1rem; letter-spacing: .25rem; margin-top: .7rem;}
  .kid-note {font-size: 1.05rem; color: #9aa6b4; text-align: center; margin-top: .8rem;}
  div[role="radiogroup"] label p {font-size: 1.25rem;}
  .stButton button p {font-size: 1.3rem; font-weight: 700;}
</style>
"""

QB_WORDS = {
    "SHOTGUN": ("Shotgun", "The quarterback stands a few big steps <b>behind</b> the line. "
                           "That gives them extra time to look for someone to throw to."),
    "PISTOL": ("Pistol", "The quarterback stands a little behind the line with a runner right behind them. "
                         "They could run <b>or</b> throw."),
    "UNDER_CENTER": ("Under center", "The quarterback is <b>right up close</b> behind the center. "
                                     "That makes it quick to hand the ball to a runner."),
}
BACKFIELD_WORDS = {
    "I_FORMATION": "Two runners are lined up behind the quarterback, one behind the other.",
    "SINGLEBACK": "One runner is waiting behind the quarterback.",
    "EMPTY": "No runner behind the quarterback. Everyone is spread out, ready to catch.",
}
SHELL_WORDS = {
    "0-HIGH": "No defenders are far back. The defense is crowding up close.",
    "1-HIGH (MOFC)": "One defender is standing far back in the middle, guarding against a long throw.",
    "2-HIGH (MOFO)": "Two defenders are standing far back, ready to stop a long throw.",
}


def out_of_ten(probability: float) -> int:
    return int(round(probability * 10))


def dots(count: int) -> str:
    return "🏈" * count + "⚪" * (10 - count)


def describe_lineup(findings: dict) -> list[str]:
    """Plain-language sentences for the confident reads only."""

    def confident(name: str):
        finding = findings.get(name)
        return finding if finding is not None and finding.known and finding.confidence >= 0.5 else None

    lines = []
    qb = confident("qb_alignment")
    if qb and qb.value in QB_WORDS:
        name, text = QB_WORDS[qb.value]
        lines.append(f"<b>{name}:</b> {text}")
    else:
        lines.append("The computer could not tell exactly where the quarterback is standing.")
    backfield = confident("backfield")
    if backfield and backfield.value in BACKFIELD_WORDS:
        lines.append(BACKFIELD_WORDS[backfield.value])
    trips = confident("trips")
    if trips and str(trips.value).startswith("TRIPS"):
        side = str(trips.value).split()[-1].lower()
        bunch = confident("bunch")
        close = " squeezed close together" if bunch and str(bunch.value).startswith("BUNCH") else ""
        lines.append(f"Three catchers are lined up on the {side} side{close}.")
    box = confident("box_count")
    if box:
        lines.append(f"<b>{box.value}</b> defenders are crowded near the middle, ready to stop a run.")
    shell = confident("safety_shell")
    if shell and shell.value in SHELL_WORDS:
        lines.append(SHELL_WORDS[shell.value])
    return lines


def show_wild_guess(base_pass_rate: Optional[float]) -> None:
    """Shown when the picture alone is not enough: fall back to how often NFL
    teams pass overall (the majority baseline), and say that plainly."""
    if base_pass_rate is None:
        fact = "most NFL plays are passes"
        dots_html = ""
    else:
        count = out_of_ten(base_pass_rate)
        fact = f"about <b>{count} out of 10</b> NFL plays are passes"
        dots_html = (f'<div class="kid-dots">{dots(count)}</div>'
                     '<div style="font-size:1.05rem; color:#9aa6b4">🏈 = pass &nbsp;&nbsp; ⚪ = run</div>')
    st.markdown(
        f"""<div class="kid-guess">
          <div style="font-size:1.3rem">The computer's best guess…</div>
          <div class="word">PASS 🎯</div>
          <div class="why">It couldn't see the players clearly enough to read this play, so it used a fun fact:
            {fact}.</div>
          {dots_html}
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="kid-note">This one is a wild guess! For a smarter guess, use a video from a wide camera '
        "that shows all the players standing still right before the snap.</div>",
        unsafe_allow_html=True,
    )


def show_guess(prediction, base_pass_rate: Optional[float]) -> None:
    if prediction is None or prediction.status != STATUS_OK:
        show_wild_guess(base_pass_rate)
        return
    call = prediction.call
    chance = prediction.p_pass if call == "PASS" else prediction.p_run
    count = out_of_ten(prediction.p_pass)
    word = "PASS 🎯" if call == "PASS" else "RUN 🏃"
    st.markdown(
        f"""<div class="kid-guess">
          <div style="font-size:1.3rem">The computer guesses…</div>
          <div class="word">{word}</div>
          <div class="why">In real NFL games, when teams lined up like this,
            about <b>{out_of_ten(chance)} out of 10</b> plays were {"passes" if call == "PASS" else "runs"}.</div>
          <div class="kid-dots" title="{count} passes out of 10">{dots(count)}</div>
          <div style="font-size:1.05rem; color:#9aa6b4">🏈 = pass &nbsp;&nbsp; ⚪ = run</div>
        </div>""",
        unsafe_allow_html=True,
    )
    accuracy = prediction.test_metrics.get("accuracy")
    if accuracy:
        st.markdown(
            f'<div class="kid-note">It is a guess, not magic! When we tested it on newer games it had never seen, '
            f"it was right about <b>{out_of_ten(accuracy)} out of 10</b> times.</div>",
            unsafe_allow_html=True,
        )


def show_play(findings: dict, overlay_bgr: Optional[np.ndarray], prediction, base_pass_rate: Optional[float]) -> None:
    st.markdown('<div class="kid-step">1. Here is what the computer saw</div>', unsafe_allow_html=True)
    if overlay_bgr is not None:
        st.image(overlay_bgr, channels="BGR", use_container_width=True)
        st.markdown(
            '<div class="kid-note">Yellow boxes = the team with the ball. Blue boxes = the defense. '
            "The yellow line is where the play starts.</div>",
            unsafe_allow_html=True,
        )
    st.markdown('<div class="kid-step">2. How the players lined up</div>', unsafe_allow_html=True)
    sentences = "".join(f"<p style='margin:.35rem 0'>{line}</p>" for line in describe_lineup(findings))
    st.markdown(f'<div class="kid-box">{sentences}</div>', unsafe_allow_html=True)
    st.markdown('<div class="kid-step">3. Run or pass?</div>', unsafe_allow_html=True)
    show_guess(prediction, base_pass_rate)


def render(
    predictor: RunPassPredictor,
    load_analyzer: Callable,
    detector: Path,
    base_pass_rate: Optional[float],
    switch_view: Callable[[], None],
) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<p class="kid-title">🏈 Guess the Play!</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="kid-sub">The computer looks at where every player stands <i>before</i> the ball is snapped, '
        "then guesses: will the team <b>run</b> or <b>pass</b>?</p>",
        unsafe_allow_html=True,
    )

    source = st.radio("Pick a play", ["Paste a YouTube link", "Upload my video", "Upload a picture"],
                      horizontal=True, key="kid_source")
    upload = None
    link = start = ""
    if source == "Paste a YouTube link":
        link = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=…", key="kid_link")
        start = st.text_input("Start watching at (minutes:seconds)", value="0:00", key="kid_start",
                              help="The computer watches one minute of the video from here.")
    elif source == "Upload my video":
        upload = st.file_uploader("Add a football video", type=["mp4", "mov", "m4v", "avi", "mkv"], key="kid_video")
    else:
        upload = st.file_uploader("Add a picture taken right before the snap", type=["png", "jpg", "jpeg"],
                                  key="kid_image")
    go = st.button("🏈 Guess the play!", type="primary", use_container_width=True, key="kid_go")

    if go:
        st.session_state.pop("result", None)
        st.session_state.pop("image_result", None)
        if source == "Upload a picture":
            if upload is None:
                st.warning("Add a picture first, then press the button.")
            else:
                image = cv2.imdecode(np.frombuffer(upload.getbuffer(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    st.warning("That file does not look like a picture. Try a PNG or JPG.")
                else:
                    with st.spinner("The computer is looking at the players… 👀"):
                        report = load_analyzer().analyze(image)
                    st.session_state["image_result"] = (image, report, predictor.predict(report.model_inputs()))
        else:
            path = None
            if source == "Paste a YouTube link":
                if not link.strip():
                    st.warning("Paste a YouTube link first, then press the button.")
                else:
                    try:
                        with st.spinner("Getting the video from YouTube… 📺"):
                            path = fetch_clip(link, start, seconds=CLIP_SECONDS)
                    except YouTubeError as error:
                        st.warning(str(error))
            elif upload is not None:
                path = Path(tempfile.mkdtemp(prefix="di_")) / Path(upload.name).name
                path.write_bytes(upload.getbuffer())
            else:
                st.warning("Add a video first, then press the button.")
            if path is not None:
                bar = st.progress(0.0, text="The computer is watching the play… 👀 (this takes about a minute)")
                try:
                    result = analyze_video(
                        path,
                        None,
                        model_path=str(detector),
                        predictor=predictor,
                        progress=lambda _stage, fraction: bar.progress(
                            min(1.0, fraction), text="The computer is watching the play… 👀"
                        ),
                    )
                    st.session_state["result"] = (str(path), result)
                except VideoOpenError:
                    st.warning("The computer could not open that video. Try an MP4 file.")
                bar.empty()

    image_result = st.session_state.get("image_result")
    video_result = st.session_state.get("result")
    if image_result is not None:
        from nfl_analytics.formation.render import draw_formation_overlay

        image, report, prediction = image_result
        if report.status != "OK":
            st.image(image, channels="BGR", use_container_width=True)
            st.markdown('<div class="kid-box">The computer could not find the players lined up in this picture.</div>',
                        unsafe_allow_html=True)
            show_wild_guess(base_pass_rate)
        else:
            show_play(report.findings, draw_formation_overlay(image, report), prediction, base_pass_rate)
    elif video_result is not None:
        _, result = video_result
        plays = result.verified_plays
        if not plays:
            st.markdown('<div class="kid-box">The computer could not find a moment where the players were lined up '
                        "and standing still.</div>", unsafe_allow_html=True)
            show_wild_guess(base_pass_rate)
        else:
            # Open on the first play the computer could make a guess for.
            index = next((n for n, play in enumerate(plays) if play.prediction.status == STATUS_OK), 0)
            if len(plays) > 1:
                labels = [f"Play {n + 1}" for n in range(len(plays))]
                index = st.radio("This video has more than one play. Pick one:", range(len(plays)), index=index,
                                 format_func=lambda n: labels[n], horizontal=True, key="kid_play")
            play = plays[index]
            show_play(play.consensus.findings, play.overlay_bgr, play.prediction, base_pass_rate)

    st.write("")
    st.divider()
    if st.button("Show me all the details (grown-up view)", key="to_detailed"):
        switch_view()
