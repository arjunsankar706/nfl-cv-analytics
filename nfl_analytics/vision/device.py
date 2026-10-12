"""Inference-device selection with graceful fallback."""

from __future__ import annotations

import warnings

import torch


def choose_device(requested: str = "auto") -> str:
    """Return an Ultralytics device string.

    ``auto`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit
    request for an unavailable accelerator falls back to CPU with a warning
    instead of failing deep inside the first inference call.
    """
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested not in ("cpu", "mps") and not torch.cuda.is_available():
        warnings.warn(f"CUDA device {requested!r} requested but unavailable; using CPU.")
        return "cpu"
    return requested
