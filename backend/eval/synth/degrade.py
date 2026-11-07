"""Image degradations that turn a clean page render into a scan or a photo (seeded, OCR-legible).

Profiles (see ``eval.labels.QUALITIES``):

``clean``       untouched render at 200 DPI.
``scan_light``  200 DPI, <= 1 degree rotation, slight blur, mild noise, JPEG quality 70.
``scan_heavy``  150 DPI, <= 3 degrees rotation, more blur and noise, speckles, a dirt streak, lost contrast, JPEG 35.
``photo``       200 DPI, uneven lighting gradient + vignette, slight perspective and rotation, a soft shadow, the page
                on a table, warm colour cast, JPEG 80.
``lowres``      rendered at 110 DPI with a touch of blur (JPEG 85 when stored as JPEG).

``degrade_image`` returns the degraded pixels and the JPEG quality the container should use (``None`` = lossless), so
the JPEG artefacts are produced exactly once, by the writer.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from eval.labels import QUALITIES

__all__ = ["DPI", "PROFILES", "QUALITIES", "degrade_image", "encode_jpeg"]

DPI: dict[str, int] = {"clean": 200, "scan_light": 200, "scan_heavy": 150, "photo": 200, "lowres": 110}


@dataclass(frozen=True)
class Profile:
    rotation: float  # maximum rotation in degrees
    blur: tuple[float, float]  # Gaussian sigma range in pixels at 200 DPI
    noise: tuple[float, float]  # sensor noise sigma range (grey levels)
    jpeg: int | None
    contrast: float = 1.0
    brightness: float = 0.0
    speckle: float = 0.0  # fraction of pixels turned into dark dust
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


PROFILES: dict[str, Profile] = {
    "clean": Profile(0.0, (0.0, 0.0), (0.0, 0.0), None),
    "scan_light": Profile(1.0, (0.45, 0.8), (2.0, 4.0), 70, 0.97, 2.0, 0.0, (1.0, 1.0, 0.99)),
    "scan_heavy": Profile(3.0, (1.3, 1.9), (8.0, 13.0), 35, 0.68, 16.0, 0.0006, (1.0, 0.985, 0.94)),
    "photo": Profile(2.2, (0.6, 1.0), (2.0, 4.0), 80, 1.0, 0.0, 0.0, (1.0, 0.97, 0.92)),
    "lowres": Profile(0.0, (0.3, 0.45), (0.0, 0.0), 85),
}

BED = (226, 226, 226)  # what shows around the paper on a scanner


def _rotate(arr: np.ndarray, angle: float, fill: tuple[int, int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    pad = int(max(h, w) * abs(np.sin(np.radians(angle))) / 2) + 6
    padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
    ph, pw = padded.shape[:2]
    matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    return cv2.warpAffine(
        padded, matrix, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill
    )


def _noise(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma)


def _scan(arr: np.ndarray, rng: np.random.Generator, p: Profile, scale: float) -> np.ndarray:
    if p.rotation:
        angle = float(rng.uniform(0.4, 1.0) * p.rotation * rng.choice((-1, 1)))
        arr = _rotate(arr, angle, BED if p.rotation > 1.5 else (240, 240, 240))
    sigma = float(rng.uniform(*p.blur)) * scale
    if sigma > 0.05:
        arr = cv2.GaussianBlur(arr, (0, 0), sigma)
    out = arr.astype(np.float32)
    out = (out - 128.0) * p.contrast + 128.0 + p.brightness
    out *= np.array(p.tint, dtype=np.float32)
    if p.noise[1] > 0:
        out += _noise(out.shape[:2], float(rng.uniform(*p.noise)), rng)[:, :, None]
    if p.speckle:
        mask = rng.random(out.shape[:2], dtype=np.float32) < p.speckle
        out[mask] = rng.uniform(40, 130)
        if rng.random() < 0.5:  # a faint vertical dirt streak from the scanner glass
            x = int(rng.integers(out.shape[1] // 8, out.shape[1] - out.shape[1] // 8))
            out[:, x : x + 2] *= 0.86
    return np.clip(out, 0, 255).astype(np.uint8)


def _lighting(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative illumination: a directional ramp, a vignette and one soft shadow (computed at low resolution)."""
    sh, sw = max(h // 8, 8), max(w // 8, 8)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    theta = float(rng.uniform(0, 2 * np.pi))
    ramp = (xs / sw - 0.5) * np.cos(theta) + (ys / sh - 0.5) * np.sin(theta)
    light = 1.0 + float(rng.uniform(0.28, 0.42)) * ramp
    radius = np.sqrt(((xs - sw / 2) / (sw / 2)) ** 2 + ((ys - sh / 2) / (sh / 2)) ** 2)
    light *= 1.0 - float(rng.uniform(0.08, 0.16)) * np.clip(radius - 0.35, 0, None)
    shadow = np.zeros((sh, sw), dtype=np.uint8)
    cx, cy = int(rng.uniform(0, sw)), int(rng.uniform(0, sh))
    axes = (max(int(sw * rng.uniform(0.2, 0.4)), 2), max(int(sh * rng.uniform(0.08, 0.2)), 2))
    cv2.ellipse(shadow, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    shadow_f = cv2.GaussianBlur(shadow.astype(np.float32) / 255.0, (0, 0), max(sh, sw) / 18)
    light *= 1.0 - float(rng.uniform(0.18, 0.3)) * shadow_f
    return cv2.resize(light.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _photo(arr: np.ndarray, rng: np.random.Generator, p: Profile, scale: float) -> np.ndarray:
    h, w = arr.shape[:2]
    margin = int(max(h, w) * rng.uniform(0.045, 0.07))
    bh, bw = h + 2 * margin, w + 2 * margin
    base = np.array(rng.choice([(92, 78, 68), (64, 64, 70), (148, 138, 126)]), dtype=np.float32)
    table = np.empty((bh, bw, 3), dtype=np.float32)
    table[:] = base
    grain = cv2.GaussianBlur(rng.standard_normal((bh // 4, bw // 4), dtype=np.float32) * 26.0, (0, 0), 2)
    table += cv2.resize(grain, (bw, bh), interpolation=cv2.INTER_LINEAR)[:, :, None]
    canvas = np.clip(table, 0, 255).astype(np.uint8)
    jitter = lambda: float(rng.uniform(-0.012, 0.012))  # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [margin + jitter() * w, margin + jitter() * h],
            [margin + w + jitter() * w, margin + jitter() * h],
            [margin + w + jitter() * w, margin + h + jitter() * h],
            [margin + jitter() * w, margin + h + jitter() * h],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    angle = float(rng.uniform(-p.rotation, p.rotation))
    rot = cv2.getRotationMatrix2D((bw / 2, bh / 2), angle, 1.0)
    warped = cv2.warpPerspective(arr, matrix, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    warped = cv2.warpAffine(warped, rot, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = warped.astype(np.float32) * _lighting(bh, bw, rng)[:, :, None] * np.array(p.tint, dtype=np.float32)
    sigma = float(rng.uniform(*p.blur)) * scale
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    out += _noise(out.shape[:2], float(rng.uniform(*p.noise)), rng)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_image(
    img: Image.Image, quality: str, rng: np.random.Generator, *, dpi: int = 200
) -> tuple[Image.Image, int | None]:
    """Degrade a rendered page; returns ``(pixels, jpeg_quality_or_None)``."""
    if quality not in PROFILES:
        raise ValueError(f"unknown quality {quality!r}")
    profile = PROFILES[quality]
    if quality == "clean":
        return img, None
    arr = np.asarray(img.convert("RGB"))
    scale = dpi / 200.0
    if quality == "photo":
        out = _photo(arr, rng, profile, scale)
    elif quality == "lowres":
        out = cv2.GaussianBlur(arr, (0, 0), float(rng.uniform(*profile.blur)) * scale) if profile.blur[1] else arr
    else:
        out = _scan(arr, rng, profile, scale)
    return Image.fromarray(out), profile.jpeg


def encode_jpeg(img: Image.Image, quality: int, *, dpi: int) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, dpi=(dpi, dpi), optimize=False)
    return buffer.getvalue()
