"""Image preparation for OCR.

Every step is conditional so that a clean scan passes through unchanged ("never make clean input worse"):

* any mode (palette, RGBA, 16-bit, CMYK ...) becomes 8-bit RGB; transparency becomes white paper;
* small images (long side < 1600 px) are upscaled to a long side of about 2000 px when the engine benefits
  (Tesseract does; RapidOCR rescales internally and reads low-resolution input *worse* after upscaling, so it
  opts out), very large ones are reduced to 4000 px (RapidOCR works at 2000 px anyway);
* low-contrast (faded) scans get a mild CLAHE contrast boost on the lightness channel;
* skew between 0.5 and 15 degrees is measured with a projection profile and straightened - this keeps a
  label and its far-right value on one OCR line once the skew exceeds a couple of degrees.

The result stays RGB: RapidOCR's models are trained on colour input.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

UPSCALE_BELOW_PX = 1600
UPSCALE_TARGET_PX = 2000
DOWNSCALE_ABOVE_PX = 4000
DESKEW_MIN_DEG = 0.5
DESKEW_MAX_DEG = 15.0
LOW_CONTRAST = 100  # difference of the mean ink and mean paper grey levels below which contrast is boosted
CLAHE_CLIP_LIMIT = 2.0

_SKEW_WORK_PX = 1000  # skew is estimated on a downscaled copy
_SKEW_COARSE_STEP = 0.5
_SKEW_FINE_STEP = 0.1
_MIN_INK, _MAX_INK = 0.002, 0.4  # outside this ink coverage the page is blank or a photo: no skew estimate

_HIGH_BIT_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I", "F"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})


def prepare_for_ocr(img: Image.Image, *, upscale: bool = True) -> np.ndarray:
    """Return an 8-bit RGB array (H x W x 3) ready for OCR; see the module docstring for the steps.

    ``upscale=False`` keeps small images at their native size (for engines that rescale internally).
    """
    rgb = _to_rgb(img)
    rgb = _rescale(rgb, upscale=upscale)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if _contrast(gray) < LOW_CONTRAST:
        rgb = _enhance_contrast(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = estimate_skew(gray)
    if DESKEW_MIN_DEG < abs(angle) < DESKEW_MAX_DEG:
        rgb = _rotate(rgb, -angle)
    return rgb


def estimate_skew(gray: np.ndarray) -> float:
    """Skew of the text in degrees, counter-clockwise positive (the convention of ``PIL.Image.rotate``).

    The ink is rotated through candidate angles and the angle whose horizontal projection profile is the
    sharpest (largest sum of squared differences between adjacent rows) wins. Returns 0.0 for blank pages and
    for images that are mostly ink (photos, inverted scans).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    factor = _SKEW_WORK_PX / max(height, width, 1)
    if factor < 1:
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coverage = float(ink.mean())
    if not _MIN_INK <= coverage <= _MAX_INK:
        return 0.0
    coarse = _sharpest_angle(ink, np.arange(-DESKEW_MAX_DEG, DESKEW_MAX_DEG + 1e-9, _SKEW_COARSE_STEP))
    fine = _sharpest_angle(
        ink, np.arange(coarse - _SKEW_COARSE_STEP, coarse + _SKEW_COARSE_STEP + 1e-9, _SKEW_FINE_STEP)
    )
    # Rotating by ``fine`` (counter-clockwise) straightens the text, so the text leans by -fine.
    return round(-fine, 2) + 0.0


def _sharpest_angle(ink: np.ndarray, angles: np.ndarray) -> float:
    height, width = ink.shape
    center = (width / 2, height / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in angles:
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.square(np.diff(profile)).sum())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def _to_rgb(img: Image.Image) -> np.ndarray:
    if img.mode in _HIGH_BIT_MODES:
        samples = np.asarray(img, dtype=np.float32)
        top = float(samples.max()) if samples.size else 0.0
        if top > 255:
            samples = samples * (255.0 / top)
        gray = np.clip(samples, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if img.mode in _ALPHA_MODES or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        paper = Image.new("RGB", rgba.size, (255, 255, 255))
        paper.paste(rgba, mask=rgba.getchannel("A"))
        return np.asarray(paper)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def _rescale(rgb: np.ndarray, *, upscale: bool) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_side = max(height, width)
    if long_side < UPSCALE_BELOW_PX:
        if not upscale:
            return rgb
        factor, interpolation = UPSCALE_TARGET_PX / long_side, cv2.INTER_CUBIC
    elif long_side > DOWNSCALE_ABOVE_PX:
        factor, interpolation = DOWNSCALE_ABOVE_PX / long_side, cv2.INTER_AREA
    else:
        return rgb
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return cv2.resize(rgb, size, interpolation=interpolation)


def _contrast(gray: np.ndarray) -> float:
    """Mean paper grey level minus mean ink grey level, splitting the two with Otsu's threshold."""
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray[gray <= threshold]
    paper = gray[gray > threshold]
    if ink.size == 0 or paper.size == 0:
        return 255.0
    return float(paper.mean() - ink.mean())


def _enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(np.ascontiguousarray(lab[:, :, 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _rotate(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees about the centre, keeping the canvas size (white fill)."""
    height, width = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        rgb, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
