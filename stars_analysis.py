"""
Stars / straight.pro-style analysis: fake satellite (as in fullhill stack), WCS, refcat, overlays.

Predicted target position ``(xpred, ypred)`` for catalog alignment uses SPICE + WCS (IDL
``lucy_finddj`` + ``ad2xy``) via :func:`lucy_spice.predicted_target_pixel_xy`. The fake-satellite
placement ``(djx, djy)`` still uses the brightness centroid (or **Define center**), matching
the stack’s use of centroid for injection while stars align to the ephemeris prediction.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFont

from fullhill import (
    centroid_bright_near_center,
    ensure_square_1024,
    fakesat_flux,
    xyshift_cubic,
)

# Session cache: same pointing → identical refcat list (saves N subprocesses for N frames).
_refcat_lock = threading.Lock()
_refcat_cache: dict[tuple[float, float, float, float], list[dict[str, float]]] = {}
_REFCAT_CACHE_MAX = 128
from fits_thumb_viewer import (
    FULL_HILL_SIZE,
    arcsec_per_pixel_from_filename,
    data_to_thumbnail_u8,
    get_image_data_and_header,
    primary_exptime_seconds,
    sky_scale,
)
from lucy_getpsf import lucy_getpsf

# Optional global PSF from "Define PSF" in FitsThumbViewer (one image with enough stars).
_STARS_PSF_OVERRIDE: np.ndarray | None = None
_STARS_PSF_SOURCE: Path | None = None


def clear_stars_psf_override() -> None:
    """Clear the PSF from **Define PSF**; fake-satellite injection stays off until a new PSF is set."""
    global _STARS_PSF_OVERRIDE, _STARS_PSF_SOURCE
    _STARS_PSF_OVERRIDE = None
    _STARS_PSF_SOURCE = None


def stars_psf_source() -> Path | None:
    """If set, Stars uses this image's PSF for every plane."""
    return _STARS_PSF_SOURCE


def set_stars_psf_from_image(path: Path) -> None:
    """Load one FITS plane, run ``lucy_getpsf``; result is reused for all ``run_stars_plane`` calls."""
    global _STARS_PSF_OVERRIDE, _STARS_PSF_SOURCE
    data, _hdr0 = get_image_data_and_header(path)
    plane = np.asarray(data, dtype=np.float64)
    if plane.ndim == 3:
        plane = np.asarray(plane[:, :, 0], dtype=np.float64)
    raw1024 = ensure_square_1024(plane)
    psf = lucy_getpsf(raw1024.copy())
    _STARS_PSF_OVERRIDE = psf
    _STARS_PSF_SOURCE = path.resolve()


def _stars_psf_override_array() -> np.ndarray | None:
    """PSF for fake-satellite injection; ``None`` if **Define PSF** has not been used this session."""
    if _STARS_PSF_OVERRIDE is None:
        return None
    return np.asarray(_STARS_PSF_OVERRIDE, dtype=np.float64)


_DEFAULT_REFCAT_EXE = "/net/eris/data1/catalogs/atlas-refcat/refcat"
_DEFAULT_REFCAT_DIR = "/net/eris/data1/catalogs/atlas-refcat/00_m_16"
_REFCAT_RECT_DEG = 0.3


def _config_refcat_paths() -> tuple[str, str] | None:
    try:
        from satsearch_config import get_config

        p = get_config().paths
        return str(p.refcat_exe), str(p.refcat_dir)
    except Exception:
        return None


def _mag_annotate_font():
    """Larger than PIL default (~10px) so magnitude labels read clearly on 1024 overlays."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, 20)
        except OSError:
            continue
    return ImageFont.load_default()


def _refcat_paths() -> tuple[str, str]:
    cfg = _config_refcat_paths()
    if cfg is not None:
        return cfg
    exe = os.environ.get("SATSEARCH_REFCAT_EXE", _DEFAULT_REFCAT_EXE)
    d = os.environ.get("SATSEARCH_REFCAT_DIR", _DEFAULT_REFCAT_DIR)
    return exe, d


def refcat_stars(ra: float, dec: float, dr_deg: float, dd_deg: float) -> list[dict[str, float]]:
    """Run atlas-refcat binary; parse lines into dicts (ra, dec, g, r, i, z, j, c, o).

    Cached by rounded (ra, dec, rect) so multiple FITS with the same field share one subprocess.
    """
    key = (round(ra, 8), round(dec, 8), round(dr_deg, 8), round(dd_deg, 8))
    with _refcat_lock:
        hit = _refcat_cache.get(key)
    if hit is not None:
        return [dict(s) for s in hit]

    exe, cat_dir = _refcat_paths()
    cmd = [
        exe,
        str(ra),
        str(dec),
        "-rect",
        f"{dr_deg},{dd_deg}",
        "-dir",
        cat_dir,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except OSError as e:
        raise RuntimeError(f"refcat failed to run ({exe}): {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"refcat exit {proc.returncode}: {err[:500]}")

    stars: list[dict[str, float]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        try:
            nums = [float(p) for p in parts if p]
        except ValueError:
            continue
        if len(nums) < 9:
            continue
        stars.append(
            {
                "ra": nums[0],
                "dec": nums[1],
                "g": nums[2],
                "r": nums[3],
                "i": nums[4],
                "z": nums[5],
                "j": nums[6],
                "c": nums[7],
                "o": nums[8],
            }
        )
    with _refcat_lock:
        if len(_refcat_cache) >= _REFCAT_CACHE_MAX:
            _refcat_cache.clear()
        _refcat_cache[key] = [dict(s) for s in stars]
    return stars


def wcs_from_path(path: Path) -> WCS:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            hdr = hdu.header
            try:
                w = WCS(hdr, naxis=2)
                if w.naxis >= 2 and ("CRVAL1" in hdr or "CTYPE1" in hdr):
                    return w
            except Exception:
                continue
        return WCS(hdul[0].header, naxis=2, relax=True)


def _native_to_display_xy(
    xn: float,
    yn: float,
    nh: int,
    nw: int,
    disp_w: int,
    disp_h: int,
) -> tuple[float, float]:
    """Map native pixel coords to displayed PIL coords (same as ``data_to_thumbnail_u8`` resize)."""
    sx = disp_w / max(float(nw), 1.0)
    sy = disp_h / max(float(nh), 1.0)
    return xn * sx, yn * sy


def _five_sigma_radius_px(skysig: float, flux: float) -> float | None:
    """IDL straight.pro: sqrt(skysig*5*sqrt(!pi)/.93*5/flux)."""
    if not np.isfinite(skysig) or not np.isfinite(flux) or flux <= 0:
        return None
    try:
        return float(math.sqrt(skysig * 5.0 * math.sqrt(math.pi) / 0.93 * 5.0 / flux))
    except (ValueError, OverflowError):
        return None


@dataclass
class StarsPlaneResult:
    path: Path
    native_display: np.ndarray
    vmin: float
    vmax: float
    pil_image: Image.Image
    star_xy_native: list[tuple[float, float]]
    star_r_mag: list[float]
    five_sigma_radius_px: float | None
    #: km per native pixel at the target (same as fullhill ``kpp``); for probe distance.
    kpp_km: float
    #: Target position in native pixels (``djx``, ``djy`` used in the pipeline).
    target_cx_native: float
    target_cy_native: float
    status_extra: str = ""  # reserved for future status fragments


def run_stars_plane(
    path: Path,
    *,
    diam_m: float,
    albedo: float,
    satdist_km: float,
    satang_deg: float,
    mag_max: float,
    mag_min: float = 5.0,
    astrometry_dx: float = 0.0,
    astrometry_dy: float = 0.0,
    target_center_native: tuple[float, float] | None = None,
) -> StarsPlaneResult:
    """Load FITS, inject fake satellite like stack, refcat + WCS, draw red circles and mag labels.

    ``astrometry_dx``, ``astrometry_dy`` are added to every refcat native pixel position (same shift
    for all stars in the plane), e.g. from interactive alignment in the Stars window.

    If ``target_center_native`` is set ``(x, y)`` in native pixels (Stars **Define center**), it
    supplies ``djx``, ``djy`` for fake-sat placement while ``xpred``, ``ypred`` still come from
    SPICE via :func:`lucy_spice.stars_ephemeris_bundle`. Refcat positions use
    ``px + djx - xpred + astrometry``, so marking the true target fixes first-order alignment vs
    ephemeris. Otherwise ``centroid_bright_near_center`` supplies ``djx``, ``djy``.

    Fake satellite injection and the 5σ / 1 m fakesat readout require a PSF from
    :func:`set_stars_psf_from_image` (thumb viewer **Define PSF**). Without it, those features are
    skipped and no per-plane ``lucy_getpsf`` is run.

    Geometry and predicted pixels come from SPICE via :func:`lucy_spice.stars_ephemeris_bundle`.
    """
    data, _hdr0 = get_image_data_and_header(path)
    plane = np.asarray(data, dtype=np.float64)
    if plane.ndim == 3:
        plane = np.asarray(plane[:, :, 0], dtype=np.float64)

    raw1024 = ensure_square_1024(plane)
    psf = _stars_psf_override_array()

    _mean, median, std_full = sigma_clipped_stats(raw1024, sigma=3.0, maxiters=5)
    skysig_full = float(std_full) if std_full is not None and std_full > 0 else 1.0
    imp = raw1024 - float(median)

    et = primary_exptime_seconds(path)
    if et is None or et <= 0:
        et = 1.0

    from lucy_spice import stars_ephemeris_bundle

    range_km, phase_deg, delta_km, xpred, ypred = stars_ephemeris_bundle(path)
    if target_center_native is not None:
        djx = float(target_center_native[0])
        djy = float(target_center_native[1])
    else:
        djx, djy = centroid_bright_near_center(imp)

    satang_rad = math.radians(satang_deg)
    satx_km = satdist_km * math.cos(satang_rad)
    saty_km = satdist_km * math.sin(satang_rad)
    asp = arcsec_per_pixel_from_filename(path)
    kpp = asp / 3600.0 * (math.pi / 180.0) * range_km

    if diam_m > 0.0 and psf is not None:
        flux_sat = fakesat_flux(diam_m, albedo, range_km, delta_km, phase_deg) * float(et)
        fake = np.zeros((1024, 1024), dtype=np.float64)
        fake[504:519, 504:519] = flux_sat * psf
        sx_pix = satx_km / kpp
        sy_pix = saty_km / kpp
        shifted = xyshift_cubic(
            fake,
            (djy - 512.5 + sy_pix),
            (djx - 512.5 + sx_pix),
        )
        imp = imp + shifted

    try:
        wcs = wcs_from_path(path)
        ra0, dec0 = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    except Exception as e:
        raise RuntimeError(f"Could not build WCS for {path.name} (extast-style astrometry): {e}") from e
    stars = refcat_stars(ra0, dec0, _REFCAT_RECT_DEG, _REFCAT_RECT_DEG)
    xs: list[float] = []
    ys: list[float] = []
    mags: list[float] = []
    for s in stars:
        rm = s["r"]
        if mag_min < rm < mag_max:
            px, py = wcs.all_world2pix(s["ra"], s["dec"], 0)
            x = float(px) + djx - xpred + astrometry_dx
            y = float(py) + djy - ypred + astrometry_dy
            xs.append(x)
            ys.append(y)
            mags.append(rm)

    _, _, vmin, vmax = sky_scale(imp)
    gray_u8 = data_to_thumbnail_u8(imp, vmin, vmax, FULL_HILL_SIZE)
    disp_w, disp_h = gray_u8.size
    rgb = gray_u8.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    nh, nw = imp.shape[:2]

    for xn, yn, rm in zip(xs, ys, mags):
        xd, yd = _native_to_display_xy(xn, yn, nh, nw, disp_w, disp_h)
        rpix = max(3.0, 8.0 * disp_w / max(nw, 1))
        x0, y0 = xd - rpix, yd - rpix
        x1, y1 = xd + rpix, yd + rpix
        draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        label = f"{rm:.1f}"
        font = _mag_annotate_font()
        if hasattr(draw, "textbbox"):
            _bx = draw.textbbox((0, 0), label, font=font)
            th = max(12, _bx[3] - _bx[1])
        else:
            th = 20
        draw.text((xd + 2, yd - th - 4), label, fill=(255, 0, 0), font=font)

    if psf is not None:
        flux_1m = fakesat_flux(1.0, albedo, range_km, delta_km, phase_deg) * float(et)
        corner = imp[900:1003, :] if imp.shape[0] > 1003 else imp
        _mc, _med_c, std_c = sigma_clipped_stats(corner, sigma=3.0, maxiters=5)
        skysig_corner = float(std_c) if std_c is not None and std_c > 0 else skysig_full
        r5 = _five_sigma_radius_px(skysig_corner, flux_1m)
    else:
        r5 = None

    pairs = list(zip(xs, ys))

    return StarsPlaneResult(
        path=path,
        native_display=imp,
        vmin=vmin,
        vmax=vmax,
        pil_image=rgb,
        star_xy_native=pairs,
        star_r_mag=mags,
        five_sigma_radius_px=r5,
        kpp_km=float(kpp),
        target_cx_native=float(djx),
        target_cy_native=float(djy),
        status_extra="",
    )
